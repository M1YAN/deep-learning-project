import argparse
import csv
from statistics import mean, stdev

import numpy as np
import torch

from .common import METHODS, PREPARED, RUNS, atomic_path, device_for, prepared_for, read_json
from .model import CatalogBeam, Generator
from .train import BEAM_WIDTH, evaluate


CUTOFFS = (5, 10, 20)
SEEDS = (17, 42, 73)
PRESETS = ("course_long", "course_wide", "course_256", "course_256_long")


def matched_test_groups(reference="course_wide"):
    alt = prepared_for(reference)
    base = read_json(PREPARED / "manifest.json")
    compared = read_json(alt / "manifest.json")
    if compared["inputs"]["base_artifacts"] != base["artifacts"]:
        raise ValueError("The comparison cache was built from a different base dataset")

    with np.load(PREPARED / "beauty.npz", allow_pickle=False) as original:
        test = original["test"]
        base_collision = original["group_sizes"][test[:, -1]] > 1
    with np.load(alt / "beauty.npz", allow_pickle=False) as expanded:
        np.testing.assert_array_equal(test, expanded["test"])
        alt_collision = expanded["group_sizes"][test[:, -1]] > 1
    only_alt = "wide_only_collision" if reference == "course_wide" else "only_256_collision"
    groups = {
        "shared_collision": base_collision & alt_collision,
        "base_only_collision": base_collision & ~alt_collision,
        only_alt: alt_collision & ~base_collision,
        "neither_collision": ~base_collision & ~alt_collision,
    }
    if reference == "course_256":
        groups["cohort_256_collision"] = alt_collision
        groups["cohort_256_singleton"] = ~alt_collision
    return test, groups


def replay(preset, rows, extra_strata, device, *, methods=METHODS):
    prepared = prepared_for(preset)
    manifest = read_json(prepared / "manifest.json")
    sizes = manifest["inputs"]["codebooks"]
    data_hash = manifest["artifacts"]["beauty.npz"]
    saved = {}
    with np.load(prepared / "beauty.npz", allow_pickle=False) as data:
        group_sizes, popularity = data["group_sizes"], data["popularity"]
        for method in methods:
            mapping = data[method]
            sids = torch.as_tensor(mapping, device=device, dtype=torch.long)
            model = Generator(codebook_sizes=sizes).to(device)
            beam = CatalogBeam(mapping, device, codebook_sizes=sizes)
            for seed in SEEDS:
                folder = RUNS / preset / method / str(seed)
                recorded = read_json(folder / "metrics.json")
                cfg = recorded["config"]
                if (cfg["preset"] != preset or cfg["method"] != method or cfg["seed"] != seed
                        or cfg["prepared_sha256"] != data_hash or cfg["codebook_sizes"] != sizes
                        or cfg["beam_width"] != BEAM_WIDTH or cfg["eval_limit"] != 0):
                    raise ValueError(f"Run configuration differs from the comparison protocol: {folder}")
                checkpoint = torch.load(folder / "best.pt", map_location=device, weights_only=True)
                if checkpoint["step"] != recorded["best_step"] or checkpoint["config"] != cfg:
                    raise ValueError(f"The best checkpoint and metrics disagree in {folder}")
                model.load_state_dict(checkpoint["state_dict"])
                test = evaluate(
                    model, rows, sids, beam, group_sizes, popularity, device,
                    cutoffs=CUTOFFS, extra_strata=extra_strata,
                )["strata"]
                for name, original in recorded["test"]["strata"].items():
                    recheck = test[name]
                    if recheck["count"] != original["count"]:
                        raise ValueError(f"Replayed target count differs for {folder}/{name}")
                    if recheck["count"]:
                        for metric in ("ndcg_at_10", "hit_at_10"):
                            np.testing.assert_allclose(
                                recheck[metric], original[metric], rtol=0, atol=1e-7,
                                err_msg=f"Replayed @10 differs for {folder}/{name}/{metric}",
                            )
                saved[method, seed] = test
            print(f"{preset}: replayed {method} at {CUTOFFS}", flush=True)
    return saved


def save_summary(preset, results, filename):
    destination = RUNS / preset / filename
    with atomic_path(destination) as temporary:
        with temporary.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "method", "test_group", "targets_per_seed", "cutoff", "seeds",
                "ndcg_mean", "ndcg_std", "hit_mean", "hit_std",
                "paired_ndcg_delta_vs_zcr_mean", "paired_ndcg_delta_vs_zcr_std",
                *[f"{metric}_seed_{seed}" for seed in SEEDS for metric in ("ndcg", "hit")],
            ])
            for method in METHODS:
                runs = [results[method, seed] for seed in SEEDS]
                zcr = [results["zcr", seed] for seed in SEEDS]
                for name, record in runs[0].items():
                    count = record["count"]
                    for k in CUTOFFS:
                        values = [(run[name][f"ndcg_at_{k}"], run[name][f"hit_at_{k}"]) for run in runs]
                        if count:
                            ndcg, hit = zip(*values)
                            delta = [value - ref[name][f"ndcg_at_{k}"] for value, ref in zip(ndcg, zcr)]
                            summary = (mean(ndcg), stdev(ndcg), mean(hit), stdev(hit), mean(delta), stdev(delta))
                        else:
                            summary = (None,) * 6
                        writer.writerow([
                            method, name, count, k, len(SEEDS), *summary,
                            *[value for pair in values for value in pair],
                        ])
    print(f"Saved {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Compare Beauty codebook sizes at cutoffs 5, 10, and 20")
    parser.add_argument("presets", nargs="+", choices=PRESETS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    device = device_for(args.device)
    compare_256 = any(preset.startswith("course_256") for preset in args.presets)
    reference = "course_256" if compare_256 else "course_wide"
    filename = "ranking_at_k_256.csv" if compare_256 else "ranking_at_k.csv"
    rows, extra_strata = matched_test_groups(reference)
    tensor = torch.as_tensor(rows, device=device, dtype=torch.long)
    for preset in args.presets:
        save_summary(preset, replay(preset, tensor, extra_strata, device), filename)


if __name__ == "__main__":
    main()
