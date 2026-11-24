from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .common import METHODS


NEIGHBOR_WEIGHT = 0.5
NEIGHBORS = 3
SWEEPS = 4


def popularity_assignment(items, preferred, popularity, size):
    available = set(range(size))
    assigned = {}
    for item in sorted(items, key=lambda i: (-popularity[i], i)):
        target = preferred[item]
        code = min(available, key=lambda code: (abs(code - target), -code))
        assigned[item] = code
        available.remove(code)
    return [assigned[item] for item in items]


def neighbor_graph(vectors, neighbors=NEIGHBORS):
    similarity = vectors @ vectors.T
    weights = np.zeros_like(similarity, dtype=np.float64)
    if len(vectors) > 1:
        np.fill_diagonal(similarity, -np.inf)
        indices = np.argsort(-similarity, kind="stable")[:, :neighbors]
        np.put_along_axis(weights, indices, np.maximum(
            np.take_along_axis(similarity, indices, axis=1), 0,
        ), axis=1)
    return np.maximum(weights, weights.T)


def neighbor_assignment(unary, distances, graph, initial, weight, *, sweeps=SWEEPS):
    """Improve the ZCR starting point by one-code moves and two-item swaps."""
    if weight == 0 or len(initial) == 1 or not graph.any():
        return initial.copy()

    codes = initial.copy()
    size = unary.shape[1]
    group_size = len(codes)
    items = np.arange(group_size)
    weights = graph * (weight * group_size / graph.sum())

    for _ in range(sweeps):
        changed = False
        for i in items:
            local = unary + weights @ distances[codes]
            current = local[items, codes]
            deltas = local[i] - current[i]
            # Independent moves double-count the disappearing edge in a swap.
            deltas[codes] += local[items, codes[i]] - current
            deltas[codes] += 2 * weights[i] * distances[codes[i], codes]
            best = int(deltas.argmin())
            if deltas[best] < -1e-10:
                owner = np.flatnonzero(codes == best)
                if owner.size:
                    codes[owner[0]] = codes[i]
                codes[i] = best
                changed = True
        if not changed:
            break
    assert len(np.unique(codes)) == group_size and np.all(codes < size)
    return codes


def build_assignments(
    prefixes, residuals, centers, embeddings, popularity, *,
    neighbor_weight=NEIGHBOR_WEIGHT, neighbors=NEIGHBORS, sweeps=SWEEPS,
):
    groups = defaultdict(list)
    for index, prefix in enumerate(map(tuple, prefixes)):
        groups[prefix].append(index)
    size = len(centers)
    max_group = max(map(len, groups.values()))
    if max_group > size:
        raise ValueError(f"Largest group has {max_group} items but only {size} last tokens")

    order = np.lexsort((np.arange(len(popularity)), -popularity))
    preferred = np.empty_like(order)
    preferred[order] = np.arange(len(order)) * size // len(order)
    unary = cdist(residuals, centers, "sqeuclidean")
    distances = cdist(centers, centers, "sqeuclidean")
    group_sizes = np.empty(len(prefixes), dtype=np.int32)
    last = {method: np.empty(len(prefixes), dtype=np.int32) for method in METHODS}
    weighted_distances = dict.fromkeys(METHODS, 0.0)
    graph_weight = 0.0

    for members in groups.values():
        items = np.asarray(members)
        group_sizes[items] = len(items)
        costs = unary[items]
        _, zcr = linear_sum_assignment(costs)
        graph = neighbor_graph(embeddings[items], neighbors)
        assignments = {
            "sequential": np.arange(len(items)),
            "global_pop": popularity_assignment(items, preferred, popularity, size),
            "zcr": zcr,
            "neighbors": neighbor_assignment(costs, distances, graph, zcr, neighbor_weight, sweeps=sweeps),
            "neighbors_zero": neighbor_assignment(costs, distances, graph, zcr, 0),
        }
        graph_weight += graph.sum()
        for method, codes in assignments.items():
            last[method][items] = codes
            weighted_distances[method] += np.sum(graph * distances[np.ix_(codes, codes)])

    sids = {method: np.column_stack((prefixes, codes)).astype(np.int32)
            for method, codes in last.items()}
    for method, codes in sids.items():
        if len(np.unique(codes, axis=0)) != len(codes):
            raise ValueError(f"{method} has duplicate semantic IDs")
    if not np.array_equal(sids["zcr"], sids["neighbors_zero"]):
        raise ValueError("The zero-weight control must equal fixed-prefix ZCR")

    stats = {
        "prefixes_used": len(groups),
        "colliding_prefixes": sum(len(group) > 1 for group in groups.values()),
        "colliding_items": int(np.count_nonzero(group_sizes > 1)),
        "collision_item_fraction": float(np.mean(group_sizes > 1)),
        "group_size_histogram": dict(sorted(Counter(map(len, groups.values())).items())),
        "largest_group": max_group,
        "last_vocab_size": size,
        "neighbor_weight": neighbor_weight,
        "neighbor_k": neighbors,
        "neighbor_sweeps": sweeps,
        "neighbors_changed_vs_zcr": int(np.count_nonzero(last["neighbors"] != last["zcr"])),
        "methods": {},
    }
    for method, codes in last.items():
        used = int(np.unique(codes).size)
        stats["methods"][method] = {
            "unique_item_ids": len(sids[method]),
            "last_tokens_used": used,
            "last_vocab_utilization": used / size,
            "mean_residual_distortion": float(unary[np.arange(len(codes)), codes].mean()),
            "mean_neighbor_centroid_distance": (
                float(weighted_distances[method] / graph_weight) if graph_weight else None
            ),
        }
    return sids, group_sizes, stats
