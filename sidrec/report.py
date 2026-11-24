import argparse
from collections import defaultdict
import csv
import statistics

from .common import RUNS, atomic_path, read_json
from .train import PRESETS


def summarize(values):
    values = [value for value in values if value is not None]
    return (
        statistics.mean(values) if values else None,
        statistics.stdev(values) if len(values) > 1 else None,
        len(values),
    )


def validation_curve(path, methods):
    with atomic_path(path) as temporary:
        with temporary.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "method", "step", "validation_group", "targets_per_seed", "seeds",
                "ndcg_at_10_mean", "ndcg_at_10_std", "hit_at_10_mean", "hit_at_10_std",
                "train_interval_loss_mean", "train_seconds_mean", "decode_seconds_mean",
            ])
            for method, runs in methods.items():
                by_step = defaultdict(list)
                for run in runs:
                    for record in run["validation"]:
                        by_step[record["step"]].append(record)
                for step, records in sorted(by_step.items()):
                    for group, info in records[0]["strata"].items():
                        scores = [record["strata"][group] for record in records]
                        ndcg, ndcg_std, _ = summarize([score["ndcg_at_10"] for score in scores])
                        hit, hit_std, _ = summarize([score["hit_at_10"] for score in scores])
                        train_loss, _, _ = summarize([record.get("loss_mean") for record in records])
                        train_time, _, _ = summarize([record.get("train_seconds") for record in records])
                        writer.writerow([
                            method, step, group, info["count"], len(records),
                            ndcg, ndcg_std, hit, hit_std, train_loss, train_time,
                            statistics.mean(record["decode_seconds"] for record in records),
                        ])


def main():
    parser = argparse.ArgumentParser(description="Aggregate Beauty metrics across seeds")
    parser.add_argument("preset", choices=PRESETS)
    args = parser.parse_args()
    folder = RUNS / args.preset
    paths = sorted(folder.glob("*/*/metrics.json"))
    if not paths:
        parser.error(f"no completed runs in {folder}")
    runs = [read_json(path) for path in paths]
    comparable = ("prepared_sha256", "steps", "eval_limit", "beam_width", "model_parameters")
    if len({tuple(run["config"][key] for key in comparable) for run in runs}) != 1:
        parser.error("metrics come from different datasets or budgets; rerun the outdated jobs")

    methods = defaultdict(list)
    for run in runs:
        methods[run["config"]["method"]].append(run)
    with atomic_path(folder / "summary.csv") as temporary:
        with temporary.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "method", "seeds", "test_group", "targets_per_seed",
                "ndcg_at_10_mean", "ndcg_at_10_std", "hit_at_10_mean", "hit_at_10_std",
                "seconds_per_train_step_mean", "full_validation_decode_seconds_mean",
                "last_tokens_used", "last_vocab_utilization",
                "paired_ndcg_delta_vs_zcr_mean", "paired_ndcg_delta_vs_zcr_std", "paired_seeds",
            ])
            zcr = {run["config"]["seed"]: run for run in methods.get("zcr", [])}
            for method, group in methods.items():
                timings = [
                    record["decode_seconds"] for run in group for record in run["validation"]
                ] if not group[0]["config"]["eval_limit"] else []
                for stratum, info in group[0]["test"]["strata"].items():
                    records = [run["test"]["strata"][stratum] for run in group]
                    ndcg, ndcg_std, _ = summarize([record["ndcg_at_10"] for record in records])
                    hit, hit_std, _ = summarize([record["hit_at_10"] for record in records])
                    paired = []
                    for run, record in zip(group, records):
                        baseline = zcr.get(run["config"]["seed"])
                        if baseline and record["ndcg_at_10"] is not None:
                            paired.append(
                                record["ndcg_at_10"] - baseline["test"]["strata"][stratum]["ndcg_at_10"]
                            )
                    codes = group[0]["codes"]
                    writer.writerow([
                        method, len(group), stratum, info["count"], ndcg, ndcg_std, hit, hit_std,
                        statistics.mean(run["train_seconds_per_step"] for run in group),
                        statistics.mean(timings) if timings else None,
                        codes["last_tokens_used"], codes["last_vocab_utilization"],
                        *summarize(paired),
                    ])
                    if stratum in ("all", "collision") and ndcg is not None:
                        print(f"{method:15s} {stratum:9s} n={info['count']:5d} "
                              f"seeds={len(group)} NDCG@10={ndcg:.5f} Hit@10={hit:.5f}")
    validation_curve(folder / "validation_curve.csv", methods)
    print(f"Saved {folder / 'summary.csv'} and {folder / 'validation_curve.csv'}")


if __name__ == "__main__":
    main()
