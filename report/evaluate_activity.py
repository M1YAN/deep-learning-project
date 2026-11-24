"""Reproduce the report's user-activity panel with the saved Beauty checkpoints."""

import argparse
import csv
from statistics import mean

import numpy as np
import torch

from sidrec.common import PREPARED_256, RUNS, atomic_path, device_for
from sidrec.evaluate import SEEDS, replay
from sidrec.prepare import load_ratings, pack, split_histories


PRESET = "course_256_long"
METHODS = ("sequential", "global_pop", "zcr", "neighbors")


def test_activity():
    histories, _ = load_ratings()
    rows, asins, _, split = split_histories(histories)
    catalog = {asin: item for item, asin in enumerate(asins)}
    activity, examples = [], []
    for events in histories.values():
        timestamp, target = events[-1]
        if timestamp <= split["test_cutoff_unix"] or target not in catalog:
            continue
        history = [catalog[asin] for _, asin in events[:-1] if asin in catalog]
        if not history:
            continue
        examples.append(pack(history, catalog[target]))
        activity.append(sum(ts <= split["validation_cutoff_unix"] for ts, _ in events))
    np.testing.assert_array_equal(examples, rows["test"])
    return np.asarray(activity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = device_for(args.device)
    activity = test_activity()
    with np.load(PREPARED_256 / "beauty.npz", allow_pickle=False) as data:
        rows = data["test"]
        collided = data["group_sizes"][rows[:, -1]] > 1
    bins = {
        "collision__activity_0_4": collided & (activity <= 4),
        "collision__activity_5_9": collided & (activity >= 5) & (activity <= 9),
        "collision__activity_10_plus": collided & (activity >= 10),
    }
    test = torch.as_tensor(rows, dtype=torch.long, device=device)
    result = replay(PRESET, test, bins, device, methods=METHODS)
    destination = RUNS / PRESET / "report_activity_at_10.csv"
    with atomic_path(destination) as temporary:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cohort", "targets_per_seed", *[f"{m}_ndcg_mean" for m in METHODS]])
            for name, mask in bins.items():
                means = [mean(result[method, seed][name]["ndcg_at_10"] for seed in SEEDS)
                         for method in METHODS]
                writer.writerow([name, int(mask.sum()), *means])
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
