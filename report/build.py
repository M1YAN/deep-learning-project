"""Generate the CVPR report figures and result tables from the experiment logs."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


REPORT = Path(__file__).resolve().parent
RUNS = REPORT.parent / "runs"
FIGURES = REPORT / "figures"
BASE = RUNS / "course_256_long"
SWEEP = RUNS / "neighbor_sweep"
SEEDS = (17, 42, 73)
CUTOFFS = (5, 10, 20)
BASELINES = {
    "sequential": "Sequential",
    "global_pop": "Global popularity",
    "zcr": "Fixed-prefix ZCR",
    "neighbors_zero": "Zero-weight control",
}
SWEEP_NAMES = {
    "neighbor_weight_025": "Neighbors (0.25, 3)",
    "default": "Neighbors (0.50, 3)",
    "neighbor_weight_100": "Neighbors (1.00, 3)",
    "neighbor_k_1": "Neighbors (0.50, 1)",
}
LABELS = {**BASELINES, **SWEEP_NAMES}
COLORS = {
    "sequential": "#4f6183",
    "global_pop": "#a75f28",
    "zcr": "#707982",
    "default": "#168678",
    "neighbor_weight_100": "#1059a1",
}
INK = "#192837"
MUTED = "#556678"
GRID = "#dbe3ea"
BLUE = "#1059a1"
TEAL = "#168678"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#a4b1be",
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "svg.fonttype": "path",
    "svg.hashsalt": "beauty-sid-course-report",
})


def records(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.svg", metadata={"Date": None}, facecolor="white")
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"CreationDate": None}, facecolor="white")
    plt.close(fig)


def pipeline():
    fig, ax = plt.subplots(figsize=(7.15, 2.18))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 7.15), ylim=(0, 2.18))
    ax.set_axis_off()

    for y, phase in ((2.03, "CATALOG  /  SID construction"), (0.99, "INTERACTIONS  /  Training and evaluation")):
        ax.text(0.04, y, phase, color=MUTED, fontsize=8, fontweight="bold", va="center")

    row_one = [
        ("Item text", "Title + description"),
        ("Text embedding", "Frozen MiniLM"),
        ("Residual K-means", "Fixed 3-code prefix"),
        ("Fourth-code assignment", "ZCR + proximity"),
        ("Unique item SID", "(code 1, 2, 3, 4)"),
    ]
    row_two = [
        ("Time split", "Warm catalog + history"),
        ("SID sequences", "Last 10 interactions"),
        ("Transformer", "Predict four codes"),
        ("Catalog beam search", "Width 20"),
        ("Item-level metrics", "All + collided targets"),
    ]
    xs = np.linspace(0.035, 5.84, 5)
    width = 1.26
    for y, row in ((1.28, row_one), (0.24, row_two)):
        for i, (title, subtitle) in enumerate(row):
            x = xs[i]
            highlighted = y == 1.28 and i == 3
            ax.add_patch(FancyBboxPatch(
                (x, y), width, 0.54,
                boxstyle="round,pad=0.012,rounding_size=0.055",
                facecolor="#e8f3fa" if highlighted else "#f3f6f8",
                edgecolor=BLUE if highlighted else "#c2ced8", linewidth=0.9,
            ))
            ax.text(x + width / 2, y + 0.35, title, fontsize=7.05, ha="center",
                    va="center", fontweight="bold")
            ax.text(x + width / 2, y + 0.16, subtitle, fontsize=6.5, color=MUTED,
                    ha="center", va="center")
            if i < len(row) - 1:
                ax.add_patch(FancyArrowPatch((x + width + 0.026, y + 0.27),
                                             (xs[i + 1] - 0.027, y + 0.27),
                                             arrowstyle="-|>", color=MUTED,
                                             linewidth=0.72, mutation_scale=7))
    ax.plot([6.47, 6.47, 2.12], [1.27, 0.885, 0.885], color=TEAL, lw=0.85, ls="--")
    ax.add_patch(FancyArrowPatch((2.12, 0.885), (2.12, 0.79), arrowstyle="-|>",
                                 color=TEAL, linewidth=0.85, mutation_scale=8))
    ax.text(4.32, 0.935, "Fixed SID mapping", color=TEAL, fontsize=6.6, ha="center")
    save(fig, "pipeline")


def validation_curves():
    rows = records(BASE / "validation_curve.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.12))
    fig.subplots_adjust(left=0.092, right=0.984, bottom=0.20, top=0.74, wspace=0.36)
    methods = ("sequential", "global_pop", "zcr", "default")
    for ax, group, metric, title in zip(
        axes, ("all", "collision", "all"),
        ("ndcg_at_10_mean", "ndcg_at_10_mean", "train_interval_loss_mean"),
        ("All targets", "Collided targets", "Training loss"),
    ):
        for method in methods:
            original = "neighbors" if method == "default" else method
            series = [r for r in rows if r["method"] == original and r["validation_group"] == group]
            series.sort(key=lambda r: int(r["step"]))
            steps = [int(r["step"]) / 1000 for r in series]
            values = [float(r[metric]) for r in series]
            ax.plot(steps, values, linewidth=1.25, color=COLORS[method], label=LABELS[method])
        ax.set(title=title, xlabel="Updates (thousands)", xticks=(5, 10, 15, 20))
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color=GRID, linewidth=0.55)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Validation NDCG@10")
    axes[1].set_ylabel("Validation NDCG@10")
    axes[2].set_ylabel("Cross-entropy")
    axes[2].legend(*axes[2].get_legend_handles_labels(), loc="lower center",
                   bbox_to_anchor=(-0.79, 1.30), ncol=4, frameon=False, fontsize=7.2)
    save(fig, "training_curves")


def read_ranking(sweep_records):
    primary = {(r["method"], r["test_group"], int(r["cutoff"])): r
               for r in records(BASE / "ranking_at_k_256.csv")}
    sweep = {(r["case"], r["group"], int(r["cutoff"])): r
             for r in sweep_records if r["split"] == "test"}
    return primary, sweep


def rank_cells(rows, formats):
    columns = []
    for values, spec in zip(zip(*rows), formats):
        top_two = sorted(set(values), reverse=True)[:2]
        styles = dict(zip(top_two, (r"\textbf", r"\underline")))
        formatted = []
        for value in values:
            number = format(value, spec).replace("%", r"\%")
            if value in styles:
                number = f"{styles[value]}{{{number}}}"
            formatted.append(number)
        columns.append(formatted)
    return list(zip(*columns))


def tex_rows(command, rows):
    body = []
    for key, cells in rows:
        shade = r"\rowcolor{neighborhighlight} " if key == "neighbor_weight_100" else ""
        body.append(shade + " & ".join(cells) + r" \\%")
    return r"\newcommand{" + command + "}{%\n" + "\n".join(body) + "\n}\n"


def ranking_table(primary, sweep):
    cohorts = [(group, cutoff) for group in ("all", "collision") for cutoff in CUTOFFS]
    metrics = ("ndcg_mean", "hit_mean")
    rows = {
        key: [float(source[key, group, cutoff][metric])
                for group, cutoff in cohorts for metric in metrics]
        for methods, source in ((BASELINES, primary), (SWEEP_NAMES, sweep))
        for key in methods
    }
    formatted = rank_cells(rows.values(), (".5f", ".3%") * len(cohorts))
    return tex_rows(r"\RankingRows", [
        (key, (LABELS[key], *[f"{ndcg} / {hit}" for ndcg, hit in zip(cells[::2], cells[1::2])]))
        for key, cells in zip(rows, formatted)
    ])


def sweep_table(sweep_records):
    scores = {(r["case"], r["split"]): r for r in sweep_records
              if r["case"] in SWEEP_NAMES and r["group"] == "collision" and r["cutoff"] == "10"}
    settings = []
    rows = []
    for case in SWEEP_NAMES:
        validation = scores[case, "validation"]
        weight, k = float(validation["neighbor_weight"]), int(validation["neighbor_k"])
        settings.append(f"$({weight:.2f}, {k})$")
        rows.append([float(scores[case, split]["ndcg_mean"]) for split in ("validation", "test")])
    return tex_rows(r"\SweepRows", [
        (case, (setting, *cells))
        for case, setting, cells in zip(SWEEP_NAMES, settings, rank_cells(rows, (".6f", ".6f")))
    ])


def paired_figure(primary, sweep):
    treatment = sweep["neighbor_weight_100", "collision", 10]
    references = ("zcr", "sequential")
    deltas = np.array([
        [1000 * (float(treatment[f"ndcg_seed_{seed}"]) -
                 float(primary[method, "collision", 10][f"ndcg_seed_{seed}"]))
         for seed in SEEDS]
        for method in references
    ])
    fig, ax = plt.subplots(figsize=(7.15, 1.91))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.77, bottom=0.25)
    ax.axvline(0, color=MUTED, linewidth=0.9)
    for i, (seed, color, marker) in enumerate(zip(SEEDS, (BLUE, TEAL, "#a75f28"), ("o", "s", "^"))):
        ax.scatter(deltas[:, i], np.array([1, 0]) + (i - 1) * 0.10,
                   marker=marker, s=34, color=color, label=f"Seed {seed}", zorder=3)
    ax.scatter(deltas.mean(1), [1.24, 0.24], marker="D", s=39, facecolors="white",
               edgecolors=INK, label="Mean", zorder=4)
    ax.set(xlim=(-6, 8), ylim=(-0.24, 1.48),
           yticks=(1, 0), yticklabels=("vs. ZCR", "vs. sequential"),
           xlabel="Paired difference in collided NDCG@10 (× 1,000)")
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=4,
              frameon=False, fontsize=7.5)
    save(fig, "paired_seeds")


def subgroup_figures(primary, sweep):
    activity_path = BASE / "report_activity_at_10.csv"
    if activity_path.exists():
        activity = {r["cohort"]: r for r in records(activity_path)}
    else:
        activity = {r["cohort"]: r for r in records(BASE / "cohort_analysis_at_k.csv")
                    if r["split"] == "test" and r["cutoff"] == "10"}
    panels = [
        ("Prefix group size", ("size_2", "size_3_4", "size_5_8", "size_9_plus"),
         ("2", "3–4", "5–8", "9+")),
        ("Target training frequency", ("collision_freq_1_5", "collision_freq_6_20", "collision_freq_21_plus"),
         ("1–5", "6–20", "21+")),
        ("User training activity", ("collision__activity_0_4", "collision__activity_5_9", "collision__activity_10_plus"),
         ("0–4", "5–9", "10+")),
    ]
    original_names = {"sequential": "sequential", "global_pop": "global_pop",
                      "zcr": "zcr", "default": "neighbors"}
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55))
    fig.subplots_adjust(left=0.078, right=0.986, bottom=0.24, top=0.73, wspace=0.31)
    handles = {}
    for i, (ax, (title, cohorts, xlabels)) in enumerate(zip(axes, panels)):
        x = np.arange(len(cohorts))
        if i == 2:
            counts = [activity[group]["targets_per_seed"] for group in cohorts]
            curves = {
                label: [float(activity[group][f"{original}_ndcg_mean"]) for group in cohorts]
                for label, original in original_names.items()
            }
        else:
            counts = [primary["zcr", group, 10]["targets_per_seed"] for group in cohorts]
            curves = {
                label: [float(primary[original, group, 10]["ndcg_mean"]) for group in cohorts]
                for label, original in original_names.items()
            }
            curves["neighbor_weight_100"] = [
                float(sweep["neighbor_weight_100", group, 10]["ndcg_mean"]) for group in cohorts
            ]
        bar_width = 0.15
        for j, (label, scores) in enumerate(curves.items()):
            offset = (j - (len(curves) - 1) / 2) * bar_width
            patch = ax.bar(x + offset, scores, width=bar_width, color=COLORS[label], label=LABELS[label])
            handles[label] = patch
        labels = [f"{label}\n(n={count})" for label, count in zip(xlabels, counts)]
        ax.set(title=title, xticks=x, xticklabels=labels, ylim=(0, 0.07))
        ax.tick_params(axis="x", length=0, labelsize=6.7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", color=GRID, linewidth=0.55)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Test NDCG@10")
    fig.legend(handles.values(), [LABELS[k] for k in handles], loc="upper center",
               bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False, fontsize=7.1)
    save(fig, "cohorts")


def sweep_figure(sweep):
    summaries = {r["case"]: r for r in records(SWEEP / "assignments.csv")}
    methods = ("zcr", "neighbor_weight_025", "default", "neighbor_weight_100")
    weights = [0, 0.25, 0.5, 1.0]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.06))
    fig.subplots_adjust(left=0.085, right=0.976, bottom=0.25, top=0.80, wspace=0.49)
    graph_metrics = [
        ("distance_on_default_graph", "Neighbor distance", "Weighted squared distance"),
        ("catalog_items_changed_vs_zcr", "Code changes", "Catalog items"),
    ]
    for ax, (key, title, ylabel) in zip(axes, graph_metrics):
        ax.plot(weights, [float(summaries[m][key]) for m in methods], "o-", color=TEAL, lw=1.45, ms=4.5)
        ax.set(title=title, ylabel=ylabel)
    test_scores = [float(sweep[m, "collision", 10]["ndcg_mean"]) for m in methods]
    axes[2].plot(weights, test_scores, "o-", color=BLUE, lw=1.45, ms=4.5)
    axes[2].set(title="Collision ranking", ylabel="Test NDCG@10")
    for ax in axes:
        ax.set(xlabel="Neighbor weight λ", xticks=weights, xticklabels=("0", ".25", ".50", "1.0"))
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color=GRID, linewidth=0.55)
        ax.set_axisbelow(True)
    save(fig, "weight_sweep")


def main():
    FIGURES.mkdir(exist_ok=True)
    sweep_records = records(SWEEP / "ranking_at_k.csv")
    primary, sweep = read_ranking(sweep_records)
    pipeline()
    validation_curves()
    paired_figure(primary, sweep)
    subgroup_figures(primary, sweep)
    sweep_figure(sweep)
    (REPORT / "ranking_rows.tex").write_text(ranking_table(primary, sweep), encoding="utf-8")
    (REPORT / "sweep_rows.tex").write_text(sweep_table(sweep_records), encoding="utf-8")


if __name__ == "__main__":
    main()
