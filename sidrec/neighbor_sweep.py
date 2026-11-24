import argparse
from collections import defaultdict
import csv
from statistics import mean, stdev

import numpy as np
import torch
from scipy.spatial.distance import cdist

from .assign import NEIGHBOR_WEIGHT, NEIGHBORS, SWEEPS, neighbor_graph
from .common import NEIGHBOR_SWEEP, PREPARED_256, RUNS, atomic_path, device_for, prepared_for, read_json
from .evaluate import CUTOFFS, SEEDS, replay


BASELINE = "course_256_long"
DEFAULT = {
    "neighbor_weight": NEIGHBOR_WEIGHT,
    "neighbors": NEIGHBORS,
    "sweeps": SWEEPS,
}
CASES = {
    "zcr": (BASELINE, "zcr", {**DEFAULT, "neighbor_weight": 0.0}),
    "default": (BASELINE, "neighbors", DEFAULT),
    **{label: (label, "neighbors", options) for label, options in NEIGHBOR_SWEEP.items()},
}


def replay_cases(rows, device):
    baseline = replay(BASELINE, rows, {}, device, methods=("zcr", "neighbors"))
    evaluations = {}
    checkpoints = {}
    for label, (preset, method, _) in CASES.items():
        replayed = baseline if preset == BASELINE else replay(preset, rows, {}, device, methods=(method,))
        seeds = []
        steps = []
        for seed in SEEDS:
            path = RUNS / preset / method / str(seed)
            metrics = read_json(path / "metrics.json")
            if metrics["config"]["steps"] != 20000:
                raise ValueError(f"Expected the 20,000-step comparison in {path}")
            best = metrics["best_step"]
            steps.append(best)
            validation = next(row["strata"] for row in metrics["validation"] if row["step"] == best)
            seeds.append({"validation": validation, "test": replayed[method, seed]})
        evaluations[label] = seeds
        checkpoints[label] = steps
    return evaluations, checkpoints


def write_ranking(destination, evaluations, checkpoints, selected):
    with atomic_path(destination) as temporary:
        with temporary.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "case", "neighbor_weight", "neighbor_k", "sweeps", "split", "group", "cutoff",
                "targets_per_seed", "seeds", "validation_selected", "ndcg_mean", "ndcg_std",
                "hit_mean", "hit_std", "paired_ndcg_delta_vs_zcr_mean", "paired_ndcg_delta_vs_zcr_std",
                *[f"{metric}_seed_{seed}" for seed in SEEDS for metric in ("best_step", "ndcg", "hit")],
            ])
            for label, runs in evaluations.items():
                options = CASES[label][2]
                for split, cutoffs in (("validation", (10,)), ("test", CUTOFFS)):
                    for group, record in runs[0][split].items():
                        count = record["count"]
                        for k in cutoffs:
                            ndcg_key, hit_key = f"ndcg_at_{k}", f"hit_at_{k}"
                            values = [(run[split][group][ndcg_key], run[split][group][hit_key]) for run in runs]
                            if count:
                                ndcg, hit = zip(*values)
                                baseline = [run[split][group][ndcg_key] for run in evaluations["zcr"]]
                                delta = [x - y for x, y in zip(ndcg, baseline)]
                                summary = (mean(ndcg), stdev(ndcg), mean(hit), stdev(hit), mean(delta), stdev(delta))
                                if k == 10 and group in ("all", "collision"):
                                    print(f"{label:20s} {split:10s} {group:9s} "
                                          f"NDCG@10={summary[0]:.6f} Hit@10={summary[2]:.6f}", flush=True)
                            else:
                                summary = (None,) * 6
                            writer.writerow([
                                label, options["neighbor_weight"], options["neighbors"], options["sweeps"],
                                split, group, k, count, len(SEEDS), label == selected, *summary,
                                *[value for step, pair in zip(checkpoints[label], values) for value in (step, *pair)],
                            ])


def write_assignment_summary(destination):
    with np.load(PREPARED_256 / "beauty.npz", allow_pickle=False) as original:
        embeddings, prefixes = original["embeddings"], original["prefixes"]
        distances = cdist(original["last_centroids"], original["last_centroids"], "sqeuclidean")
        targets = original["test"][:, -1]
        fixed, default = original["zcr"][:, -1], original["neighbors"][:, -1]
    groups = defaultdict(list)
    for item, prefix in enumerate(map(tuple, prefixes)):
        groups[prefix].append(item)
    graphs = [(items, neighbor_graph(embeddings[items], NEIGHBORS))
              for items in groups.values() if len(items) > 1]
    total_weight = sum(graph.sum() for _, graph in graphs)

    with atomic_path(destination) as temporary:
        with temporary.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "case", "catalog_items_changed_vs_zcr", "catalog_items_changed_vs_default",
                "test_targets_changed_vs_zcr", "distance_on_default_graph", "mean_residual_distortion",
            ])
            for label, (preset, method, _) in CASES.items():
                prepared = prepared_for(preset)
                with np.load(prepared / "beauty.npz", allow_pickle=False) as archive:
                    codes = archive[method][:, -1]
                stats = read_json(prepared / "manifest.json")["codes"]["methods"][method]
                weighted = sum(np.sum(graph * distances[np.ix_(codes[items], codes[items])])
                               for items, graph in graphs)
                writer.writerow([
                    label, np.count_nonzero(codes != fixed), np.count_nonzero(codes != default),
                    np.count_nonzero(codes[targets] != fixed[targets]), weighted / total_weight,
                    stats["mean_residual_distortion"],
                ])


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Beauty 256-codebook neighbor-parameter sweep")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    device = device_for(args.device)
    with np.load(PREPARED_256 / "beauty.npz", allow_pickle=False) as archive:
        rows = torch.as_tensor(archive["test"], device=device, dtype=torch.long)
    evaluations, checkpoints = replay_cases(rows, device)

    def validation_score(label):
        return tuple(mean(row["validation"][group]["ndcg_at_10"] for row in evaluations[label])
                     for group in ("collision", "all"))

    selected = max(evaluations, key=validation_score)
    output = RUNS / "neighbor_sweep"
    write_ranking(output / "ranking_at_k.csv", evaluations, checkpoints, selected)
    write_assignment_summary(output / "assignments.csv")
    print(f"Best mean validation NDCG@10 (collision first): {selected}")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
