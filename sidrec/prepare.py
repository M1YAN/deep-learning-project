import argparse
import ast
from collections import Counter, defaultdict
import csv
import gzip
import html
import json
import re
from time import perf_counter

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from .assign import build_assignments
from .common import (
    CODEBOOK_SIZES, DATA, HISTORY_LENGTH, PREPARED, atomic_path, device_for,
    CODEBOOK_SIZES_256, METHODS, NEIGHBOR_SWEEP, PREPARED_256, WIDE_CODEBOOK_SIZES, WIDE_PREPARED,
    prepared_for, read_json, sha256, write_json,
)


RATINGS = DATA / "raw" / "Beauty" / "ratings_Beauty.csv"
METADATA = DATA / "raw" / "Beauty" / "meta_Beauty.json.gz"
MINILM = DATA / "models" / "all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MINILM_FILES = (
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.txt", "model.safetensors",
)
TEXT_LENGTH = 128
QUANTIZER_SEED = 2026
TAG = re.compile(r"</?[a-zA-Z][^>]*>")
CACHE_VERSION = 1
PREPARED_FILES = ("beauty.npz", "items.json")


def load_ratings():
    histories = defaultdict(dict)
    raw_count = 0
    with RATINGS.open(newline="") as handle:
        for user, asin, _, timestamp in csv.reader(handle):
            raw_count += 1
            timestamp = int(timestamp)
            previous = histories[user].get(asin)
            if previous is None or timestamp < previous:
                histories[user][asin] = timestamp
    unique_count = sum(map(len, histories.values()))

    # Iteratively prune and count distinct user-item pairs.
    while True:
        histories = {user: events for user, events in histories.items() if len(events) >= 5}
        item_counts = Counter(asin for events in histories.values() for asin in events.keys())
        if all(n >= 5 for n in item_counts.values()):
            break
        histories = {
            user: {asin: ts for asin, ts in events.items() if item_counts[asin] >= 5}
            for user, events in histories.items()
        }
    ordered = {
        user: sorted((ts, asin) for asin, ts in events.items())
        for user, events in sorted(histories.items())
    }
    stats = {
        "raw_reviews": raw_count,
        "unique_user_item_pairs": unique_count,
        "users_5core": len(ordered),
        "items_5core": len(item_counts),
        "interactions_5core": sum(map(len, ordered.values())),
    }
    return ordered, stats


def pack(history, target):
    history = history[-HISTORY_LENGTH:]
    return [-1] * (HISTORY_LENGTH - len(history)) + history + [target]


def split_histories(histories):
    dates = np.fromiter((ts for events in histories.values() for ts, _ in events), dtype=np.int64)
    test_cut = int(np.quantile(dates, 0.90))
    val_cut = int(np.quantile(dates[dates <= test_cut], 0.90))

    train_sequences = []
    for events in histories.values():
        earlier = [asin for ts, asin in events if ts <= val_cut]
        if len(earlier) >= 2:
            train_sequences.append(earlier)

    counts = Counter(asin for seq in train_sequences for asin in seq)
    asins = sorted(counts)
    item_ids = {asin: i for i, asin in enumerate(asins)}
    popularity = np.array([counts[asin] for asin in asins], dtype=np.int64)
    rows = {"train": [], "validation": [], "test": []}
    for sequence in train_sequences:
        encoded = [item_ids[asin] for asin in sequence]
        rows["train"].extend(
            pack(encoded[max(0, i - HISTORY_LENGTH):i], encoded[i]) for i in range(1, len(encoded))
        )

    holdout_stats = {}
    for name, start, end in (("validation", val_cut, test_cut), ("test", test_cut, float("inf"))):
        cold_target = empty_history = candidate_users = 0
        for events in histories.values():
            sequence = [(ts, asin) for ts, asin in events if ts <= end]
            if not sequence or sequence[-1][0] <= start:
                continue
            candidate_users += 1
            # Select the real last target first so warm-only filtering cannot change it.
            target = item_ids.get(sequence[-1][1])
            if target is None:
                cold_target += 1
                continue
            history = [item_ids[asin] for _, asin in sequence[:-1] if asin in item_ids]
            if not history:
                empty_history += 1
                continue
            rows[name].append(pack(history, target))
        holdout_stats[name] = {
            "candidate_users": candidate_users,
            "cold_targets_dropped": cold_target,
            "empty_warm_history_dropped": empty_history,
        }
    arrays = {key: np.asarray(value, dtype=np.int32).reshape(-1, HISTORY_LENGTH + 1)
              for key, value in rows.items()}
    if any(len(array) == 0 for array in arrays.values()):
        raise ValueError("The global split produced an empty partition")
    stats = {
        "validation_cutoff_unix": val_cut,
        "test_cutoff_unix": test_cut,
        "train_users": len(train_sequences),
        "train_interactions": int(popularity.sum()),
        "catalog_size": len(asins),
        "examples": {key: len(value) for key, value in arrays.items()},
        "holdout_filtering": holdout_stats,
    }
    return arrays, asins, popularity, stats


def clean_text(value):
    if isinstance(value, list):
        value = " ".join(part for part in value if isinstance(part, str))
    if not isinstance(value, str):
        return ""
    return " ".join(TAG.sub(" ", html.unescape(value)).split())


def product_text(asins):
    requested = set(asins)
    text_by_asin = {}
    bad_lines = 0
    with gzip.open(METADATA, mode="rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                try:
                    entry = ast.literal_eval(line)
                except (SyntaxError, ValueError):
                    bad_lines += 1
                    continue
            if not isinstance(entry, dict) or entry.get("asin") not in requested:
                continue
            text = " ".join(filter(None, (
                clean_text(entry.get("title")), clean_text(entry.get("description")),
            )))
            if text:
                text_by_asin[entry["asin"]] = text
            if len(text_by_asin) == len(requested):
                break
    return [text_by_asin.get(asin, "Beauty product") for asin in asins], {
        "items_without_text": len(asins) - len(text_by_asin),
        "unparseable_metadata_lines": bad_lines,
    }


@torch.inference_mode()
def embed(texts, device, batch_size):
    model = AutoModel.from_pretrained(MINILM, local_files_only=True, use_safetensors=True)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MINILM, local_files_only=True)
    batches = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size], padding=True, truncation=True,
            max_length=TEXT_LENGTH, return_tensors="pt",
        ).to(device)
        hidden = model(**encoded).last_hidden_state
        mask = encoded.attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        batches.append(F.normalize(pooled, dim=-1).cpu().numpy())
    return np.concatenate(batches)


def quantize(embeddings, threads, codebook_sizes=CODEBOOK_SIZES):
    residual = embeddings.copy()
    prefixes = []
    with threadpool_limits(limits=threads):
        for level, size in enumerate(codebook_sizes):
            cluster = KMeans(
                n_clusters=size, n_init=3, max_iter=100, random_state=QUANTIZER_SEED + level,
            ).fit(residual)
            if level == len(codebook_sizes) - 1:
                return np.column_stack(prefixes), residual, cluster.cluster_centers_
            prefixes.append(cluster.labels_)
            residual -= cluster.cluster_centers_[cluster.labels_]
    raise AssertionError("Missing fourth-level codebook")


def prepared_matches(destination, inputs):
    manifest = destination / "manifest.json"
    if not manifest.exists():
        return False
    old = read_json(manifest)
    return old["inputs"] == inputs and all(
        (destination / name).exists() and sha256(destination / name) == old["artifacts"][name]
        for name in PREPARED_FILES
    )


def write_quantized(destination, inputs, data, asins, threads, metadata, started):
    print(f"Fitting residual K-means with codebooks {inputs['codebooks']}", flush=True)
    fitted = quantize(data["embeddings"], threads, inputs["codebooks"])
    write_assignments(destination, inputs, data, asins, fitted, metadata, started)


def write_assignments(destination, inputs, data, asins, fitted, metadata, started):
    embeddings, popularity = data["embeddings"], data["popularity"]
    prefixes, residual, centers = fitted
    sids, group_sizes, sid_stats = build_assignments(
        prefixes, residual, centers, embeddings, popularity, **inputs.get("neighbor_options", {}),
    )
    splits = {split: data[split] for split in ("train", "validation", "test")}
    for split, rows in splits.items():
        targets = rows[:, -1]
        sid_stats[f"{split}_collision_target_count"] = int(np.count_nonzero(group_sizes[targets] > 1))
        sid_stats[f"{split}_collision_target_fraction"] = float(np.mean(group_sizes[targets] > 1))

    with atomic_path(destination / "beauty.npz") as temporary:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle, **splits, **sids, popularity=popularity, group_sizes=group_sizes,
                embeddings=embeddings, prefixes=prefixes, last_centroids=centers,
            )
    write_json(destination / "items.json", asins)
    write_json(destination / "manifest.json", {
        **metadata,
        "inputs": inputs,
        "artifacts": {name: sha256(destination / name) for name in PREPARED_FILES},
        "codes": sid_stats,
        "seconds": {**metadata["seconds"], "total": perf_counter() - started},
    })
    print(f"Prepared {destination / 'manifest.json'}", flush=True)


def requantize(destination, codebooks, threads, force, started):
    base = read_json(PREPARED / "manifest.json")
    if not prepared_matches(PREPARED, base["inputs"]):
        raise ValueError("The base Beauty cache is incomplete; prepare it before requantizing")
    inputs = {
        **base["inputs"], "codebooks": list(codebooks),
        "base_artifacts": base["artifacts"],
    }
    if not force and prepared_matches(destination, inputs):
        print(f"Reusing {destination}", flush=True)
        return
    with np.load(PREPARED / "beauty.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in ("train", "validation", "test", "embeddings", "popularity")}
    write_quantized(
        destination, inputs, data, read_json(PREPARED / "items.json"), threads,
        {"data": base["data"], "device": "cpu", "seconds": {}}, started,
    )

def prepare_neighbor_sweep(force, started):
    base = read_json(PREPARED_256 / "manifest.json")
    if not prepared_matches(PREPARED_256, base["inputs"]):
        raise ValueError("The original 256-codebook cache is incomplete")
    pending = {}
    for preset, options in NEIGHBOR_SWEEP.items():
        destination = prepared_for(preset)
        inputs = {**base["inputs"], "reference_artifacts": base["artifacts"], "neighbor_options": options}
        if force or not prepared_matches(destination, inputs):
            pending[destination] = inputs
    if not pending:
        print("Reusing all neighbor-sweep code tables", flush=True)
        return

    with np.load(PREPARED_256 / "beauty.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in ("train", "validation", "test", "embeddings", "popularity")}
        # The original 256-codebook K-means run used two threads.
        fitted = quantize(data["embeddings"], 2, base["inputs"]["codebooks"])
        np.testing.assert_array_equal(fitted[0], archive["prefixes"])
        np.testing.assert_array_equal(fitted[2], archive["last_centroids"])
        reference, _, _ = build_assignments(*fitted, data["embeddings"], data["popularity"])
        for method in METHODS:
            np.testing.assert_array_equal(reference[method], archive[method])
    asins = read_json(PREPARED_256 / "items.json")
    for destination, inputs in pending.items():
        write_assignments(
            destination, inputs, data, asins, fitted,
            {"data": base["data"], "device": "cpu", "seconds": {}}, started,
        )


def main():
    parser = argparse.ArgumentParser(description="Prepare shared Beauty data and fixed semantic IDs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--embed-batch", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    size_args = parser.add_mutually_exclusive_group()
    size_args.add_argument("--wide", action="store_true", help="fit the 64/32/16/64 codebooks")
    size_args.add_argument("--prefix-256", action="store_true", help="fit the 256/256/256/64 codebooks")
    size_args.add_argument("--neighbor-sweep", action="store_true", help="prepare the neighbor parameter sweep")
    args = parser.parse_args()
    if min(args.threads, args.embed_batch) < 1:
        parser.error("threads and embed-batch must be positive")
    torch.set_num_threads(args.threads)
    started = perf_counter()
    if args.neighbor_sweep:
        if not (PREPARED_256 / "manifest.json").is_file():
            parser.error("prepare the original 256-codebook cache before the neighbor sweep")
        prepare_neighbor_sweep(args.force, started)
        return
    if args.wide or args.prefix_256:
        if not (PREPARED / "manifest.json").is_file():
            parser.error("prepare the base Beauty cache before requantizing")
        destination, sizes = (WIDE_PREPARED, WIDE_CODEBOOK_SIZES) if args.wide else (
            PREPARED_256, CODEBOOK_SIZES_256,
        )
        requantize(destination, sizes, args.threads, args.force, started)
        return

    files = [RATINGS, METADATA, *(MINILM / name for name in MINILM_FILES)]
    if any(not path.is_file() for path in files):
        parser.error("missing source files; run bash scripts/download_beauty.sh")
    inputs = {
        "sha256": {str(path.relative_to(DATA)): sha256(path) for path in files},
        "minilm_revision": MINILM_REVISION,
        "text_length": TEXT_LENGTH,
        "codebooks": list(CODEBOOK_SIZES),
        "history_length": HISTORY_LENGTH,
        "quantizer_seed": QUANTIZER_SEED,
        "cache_version": CACHE_VERSION,
    }
    if not args.force and prepared_matches(PREPARED, inputs):
        print(f"Reusing {PREPARED}", flush=True)
        return

    device = device_for(args.device)
    histories, raw_stats = load_ratings()
    splits, asins, popularity, split_stats = split_histories(histories)
    texts, text_stats = product_text(asins)
    print(f"Beauty {split_stats['examples']}; encoding {len(asins)} items on {device}", flush=True)
    embeddings = embed(texts, device, args.embed_batch)
    embed_finished = perf_counter()
    write_quantized(
        PREPARED, inputs, {**splits, "embeddings": embeddings, "popularity": popularity}, asins,
        args.threads, {
            "data": {**raw_stats, **split_stats, **text_stats}, "device": str(device),
            "seconds": {"through_embedding": embed_finished - started},
        }, started,
    )


if __name__ == "__main__":
    main()
