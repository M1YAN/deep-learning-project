import argparse
from datetime import datetime, timezone
import fcntl
import json
from math import isfinite
from shutil import copyfile
from time import perf_counter

import numpy as np
import torch

from .common import (
    METHODS, NEIGHBOR_SWEEP, RUNS, atomic_path, device_for, prepared_for, read_json,
    synchronize, write_json,
)
from .model import CatalogBeam, Generator


PRESETS = {
    "smoke": {"steps": 4, "validate_every": 4, "eval_limit": 128},
    "pilot": {"steps": 300, "validate_every": 300, "eval_limit": 0},
    "course": {"steps": 1500, "validate_every": 500, "eval_limit": 0},
    "course_long": {"steps": 10000, "validate_every": 500, "eval_limit": 0},
    "course_wide": {"steps": 10000, "validate_every": 500, "eval_limit": 0},
    "course_256": {"steps": 10000, "validate_every": 500, "eval_limit": 0},
    "course_256_long": {"steps": 20000, "validate_every": 500, "eval_limit": 0},
}
PRESETS.update({preset: PRESETS["course_256_long"] for preset in NEIGHBOR_SWEEP})
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
BEAM_WIDTH = 20
TOP_K = 10
LEARNING_RATE = 0.001
LOG_EVERY = 100
CONTINUE_FROM = {"course_256_long": "course_256"}


def strata(group_sizes, popularity):
    collision = group_sizes > 1
    return {
        "all": np.ones(len(group_sizes), dtype=bool),
        "collision": collision,
        "size_1": group_sizes == 1,
        "size_2": group_sizes == 2,
        "size_3_4": (group_sizes >= 3) & (group_sizes <= 4),
        "size_5_8": (group_sizes >= 5) & (group_sizes <= 8),
        "size_9_plus": group_sizes >= 9,
        "freq_1_5": popularity <= 5,
        "freq_6_20": (popularity >= 6) & (popularity <= 20),
        "freq_21_plus": popularity >= 21,
        "collision_freq_1_5": collision & (popularity <= 5),
        "collision_freq_6_20": collision & (popularity >= 6) & (popularity <= 20),
        "collision_freq_21_plus": collision & (popularity >= 21),
    }


@torch.inference_mode()
def evaluate(model, rows, sids, beam, group_sizes, popularity, device, *, cutoffs=(TOP_K,), extra_strata=None):
    model.eval()
    synchronize(device)
    started = perf_counter()
    by_cutoff = {cutoff: ([], []) for cutoff in cutoffs}
    for batch in rows.split(EVAL_BATCH_SIZE):
        items = beam.rank(model, batch, sids, BEAM_WIDTH, max(cutoffs))
        correct = items.eq(batch[:, -1:])
        ranks = torch.arange(2, 2 + items.shape[1], dtype=torch.float32, device=device)
        gains = correct / ranks.log2()
        for cutoff, (hits, discounts) in by_cutoff.items():
            hits.append(correct[:, :cutoff].any(dim=1))
            discounts.append(gains[:, :cutoff].sum(dim=1))
    synchronize(device)
    seconds = perf_counter() - started
    targets = rows[:, -1].cpu().numpy()
    test_strata = strata(group_sizes[targets], popularity[targets])
    if extra_strata:
        test_strata.update(extra_strata)
    results = {name: {"count": int(mask.sum())} for name, mask in test_strata.items()}
    for cutoff, (hits, discounts) in by_cutoff.items():
        hits = torch.cat(hits).cpu().numpy()
        discounts = torch.cat(discounts).cpu().numpy()
        for name, mask in test_strata.items():
            count = results[name]["count"]
            results[name][f"hit_at_{cutoff}"] = float(hits[mask].mean()) if count else None
            results[name][f"ndcg_at_{cutoff}"] = float(discounts[mask].mean()) if count else None
    return {"strata": results, "decode_seconds": seconds, "examples": len(rows)}


def make_batch_order(length, seed):
    rng = np.random.default_rng(seed)
    while True:
        permutation = rng.permutation(length)
        for start in range(0, length, BATCH_SIZE):
            yield permutation[start:start + BATCH_SIZE]


def append_record(path, record):
    with path.open("a") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")


def validation_score(metrics):
    groups = metrics["strata"]
    overall = groups["all"]["ndcg_at_10"]
    collision = groups["collision"]["ndcg_at_10"]
    return (overall if collision is None else collision, overall)


def restore_training(path, config, model, optimizer, batches, device):
    source = RUNS / CONTINUE_FROM[config["preset"]] / config["method"] / str(config["seed"])
    if not (source / "training_state.pt").is_file():
        raise ValueError(f"Complete the original training run before continuation: {source}")
    with (source / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        previous = read_json(source / "metrics.json")
        original = previous["config"]
        compared = ("prepared_sha256", "codebook_sizes", "eval_limit", "device", "device_name",
                    "torch", "train_batch", "beam_width", "learning_rate", "validate_every")
        if any(original[key] != config[key] for key in compared):
            raise ValueError(f"Continuation requires the original data and training settings: {source}")

        state = torch.load(source / "training_state.pt", map_location="cpu", weights_only=True)
        if state["config"] != original or state["step"] != original["steps"] or state["step"] >= config["steps"]:
            raise ValueError(f"Invalid continuation checkpoint: {source}")
        records = previous["validation"]
        for record in records:
            relative = record["checkpoint"]
            with atomic_path(path / relative) as temporary:
                copyfile(source / relative, temporary)
        for name in ("train_steps.jsonl", "validation.jsonl"):
            with atomic_path(path / name) as temporary:
                copyfile(source / name, temporary)
        best = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
        with atomic_path(path / "best.pt") as temporary:
            torch.save({**best, "config": config}, temporary)

    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    for _ in range(state["step"]):
        next(batches)
    torch.set_rng_state(state["cpu_rng"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
    print(f"Continuing {source} from step {state['step']}", flush=True)
    return previous, state["step"], state["examples_seen"]


def train(args, path, prepared):
    settings = dict(PRESETS[args.preset])
    if args.steps is not None:
        settings["steps"] = args.steps
        settings["validate_every"] = min(settings["validate_every"], args.steps)
    if args.eval_limit is not None:
        settings["eval_limit"] = args.eval_limit
    device = device_for(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    manifest = read_json(prepared / "manifest.json")
    codebook_sizes = manifest["inputs"]["codebooks"]

    with np.load(prepared / "beauty.npz", allow_pickle=False) as data:
        mapping = data[args.method]
        sids = torch.as_tensor(mapping, dtype=torch.long, device=device)
        group_sizes, popularity = data["group_sizes"], data["popularity"]
        train_rows = torch.as_tensor(data["train"], dtype=torch.long, device=device)
        holdouts = {}
        for index, split in enumerate(("validation", "test")):
            rows = data[split]
            if 0 < settings["eval_limit"] < len(rows):
                chosen = np.random.default_rng(2026 + index).choice(
                    len(rows), size=settings["eval_limit"], replace=False,
                )
                rows = rows[np.sort(chosen)]
            holdouts[split] = torch.as_tensor(rows, dtype=torch.long, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = Generator(codebook_sizes=codebook_sizes).to(device)
    beam = CatalogBeam(mapping, device, codebook_sizes=codebook_sizes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    batches = make_batch_order(len(train_rows), args.seed)
    config = {
        "preset": args.preset,
        "method": args.method,
        "seed": args.seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch": str(torch.__version__),
        "codebook_sizes": codebook_sizes,
        "prepared_sha256": manifest["artifacts"]["beauty.npz"],
        "train_batch": BATCH_SIZE,
        "train_examples": len(train_rows),
        "log_every": LOG_EVERY,
        "eval_batch": EVAL_BATCH_SIZE,
        "beam_width": BEAM_WIDTH,
        "learning_rate": LEARNING_RATE,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        **settings,
    }
    if args.preset in NEIGHBOR_SWEEP:
        config["neighbor_options"] = manifest["inputs"]["neighbor_options"]
    print(json.dumps(config), flush=True)
    best_score = (-1.0, -1.0)
    best_step = 0
    validations = []
    training_seconds = 0.0
    checkpoint = path / "best.pt"
    step_log = path / "train_steps.jsonl"
    validation_log = path / "validation.jsonl"
    loss_totals = torch.zeros(len(codebook_sizes), device=device)
    examples_seen = interval_examples = previous_step = 0
    if args.preset in CONTINUE_FROM:
        previous, previous_step, examples_seen = restore_training(
            path, config, model, optimizer, batches, device,
        )
        validations = previous["validation"]
        best_step = previous["best_step"]
        best_score = validation_score(next(record for record in validations if record["step"] == best_step))
        training_seconds = previous["train_seconds"]
    else:
        step_log.write_text("")
        validation_log.write_text("")
    training_start = perf_counter()
    for step in range(previous_step + 1, settings["steps"] + 1):
        model.train()
        batch = train_rows[torch.as_tensor(next(batches), device=device)]
        optimizer.zero_grad(set_to_none=True)
        level_losses = model.loss(batch, sids, per_level=True)
        loss = level_losses.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_totals.add_(level_losses.detach(), alpha=len(batch))
        interval_examples += len(batch)
        examples_seen += len(batch)
        validate = step % settings["validate_every"] == 0 or step == settings["steps"]
        if not validate and step % LOG_EVERY:
            continue

        synchronize(device)
        training_seconds += perf_counter() - training_start
        window_losses = (loss_totals / interval_examples).tolist()
        if not all(isfinite(value) for value in window_losses):
            raise RuntimeError(f"Non-finite training loss in steps {previous_step + 1}–{step}")
        progress = {
            "start_step": previous_step + 1,
            "step": step,
            "examples_in_interval": interval_examples,
            "examples_seen": examples_seen,
            "epoch_equivalent": examples_seen / len(train_rows),
            "loss_mean": sum(window_losses) / len(window_losses),
            "loss_by_level": window_losses,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_seconds": training_seconds,
        }
        append_record(step_log, progress)
        loss_totals.zero_()
        interval_examples = 0
        previous_step = step

        if validate:
            metrics = evaluate(model, holdouts["validation"], sids, beam, group_sizes, popularity, device)
            score = validation_score(metrics)
            overall = metrics["strata"]["all"]["ndcg_at_10"]
            collision = metrics["strata"]["collision"]["ndcg_at_10"]
            is_best = score > best_score

            snapshot = path / "checkpoints" / f"step-{step:06d}.pt"
            payload = {"state_dict": model.state_dict(), "config": config, "step": step}
            with atomic_path(snapshot) as temporary:
                torch.save(payload, temporary)
            if is_best:
                best_score, best_step = score, step
                with atomic_path(checkpoint) as temporary:
                    torch.save(payload, temporary)

            record = {
                **progress, **metrics, "loss": loss.item(), "best_so_far": is_best,
                "checkpoint": str(snapshot.relative_to(path)),
            }
            validations.append(record)
            append_record(validation_log, record)
            print(json.dumps({
                "step": step, "loss_mean": progress["loss_mean"],
                "train_seconds": round(training_seconds, 2),
                "val_all_ndcg": overall, "val_collision_ndcg": collision,
                "val_decode_seconds": round(metrics["decode_seconds"], 2), "best_so_far": is_best,
            }), flush=True)
        training_start = perf_counter()

    if args.preset == "course_256":
        with atomic_path(path / "training_state.pt") as temporary:
            torch.save({
                "config": config, "step": step, "examples_seen": examples_seen,
                "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            }, temporary)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
    test = evaluate(model, holdouts["test"], sids, beam, group_sizes, popularity, device)
    result = {
        "config": config,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "best_step": best_step,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "train_seconds": training_seconds,
        "train_seconds_per_step": training_seconds / settings["steps"],
        "validation": validations,
        "test": test,
        "codes": manifest["codes"]["methods"][args.method],
    }
    write_json(path / "metrics.json", result)
    print(f"Saved {path / 'metrics.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train one Beauty model and evaluate item-level ranking")
    parser.add_argument("preset", choices=PRESETS)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--force", action="store_true", help="rerun even if metrics already exist")
    args = parser.parse_args()
    if args.threads < 1 or args.seed < 0 or (args.steps is not None and args.steps < 1):
        parser.error("threads and steps must be positive, and seed must be nonnegative")
    if args.eval_limit is not None and args.eval_limit < 0:
        parser.error("eval-limit must be nonnegative")
    prepared = prepared_for(args.preset)
    if not (prepared / "manifest.json").is_file():
        parser.error(f"prepared data are missing at {prepared}; use scripts/run_experiments.sh")
    run_name = args.preset
    if args.steps is not None or args.eval_limit is not None:
        run_name += f"-steps-{args.steps}-eval-{args.eval_limit}"
    path = RUNS / run_name / args.method / str(args.seed)
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = path / "metrics.json"
        data_hash = read_json(prepared / "manifest.json")["artifacts"]["beauty.npz"]
        if current.is_file():
            matches = read_json(current)["config"]["prepared_sha256"] == data_hash
            can_resume = args.preset != "course_256" or (path / "training_state.pt").is_file()
            if not args.force and matches and can_resume and (path / "best.pt").is_file():
                print(f"Reusing completed run {path}", flush=True)
                return
            current.unlink()
        train(args, path, prepared)


if __name__ == "__main__":
    main()
