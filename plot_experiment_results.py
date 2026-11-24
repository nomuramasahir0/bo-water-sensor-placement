"""Create publication-ready figures for the fixed 30-seed experiment."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import yaml

COLORS = {
    "Random search": "#7F7F7F",
    "Local search": "#E69F00",
    "GraphComBO": "#0072B2",
    "Proposed": "#009E73",
}
LINESTYLES = {
    "Random search": ":",
    "Local search": "--",
    "GraphComBO": "-.",
    "Proposed": "-",
}


def load_rows(path: Path, method: str | None = None):
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if method is not None:
        rows = [row for row in rows if row.get("method") == method]
    return rows


def trajectories(rows, seeds, budget):
    lookup = {
        (int(row["seed"]), int(row["query"])): float(row["objective_hours"])
        for row in rows
    }
    objectives = np.asarray(
        [[lookup[seed, query] for query in range(1, budget + 1)] for seed in seeds]
    )
    return np.minimum.accumulate(objectives, axis=1)


def style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def save(fig, directory: Path, stem: str):
    fig.savefig(directory / f"{stem}.pdf")
    fig.savefig(directory / f"{stem}.png", dpi=300)
    plt.close(fig)


def plot_best_so_far(data, output):
    fig, ax = plt.subplots(figsize=(5.25, 3.25))
    queries = np.arange(1, next(iter(data.values())).shape[1] + 1)
    for label, values in data.items():
        median = np.median(values, axis=0)
        q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
        ax.fill_between(queries, q25, q75, color=COLORS[label], alpha=0.13, linewidth=0)
        ax.plot(
            queries,
            median,
            label=label,
            color=COLORS[label],
            linestyle=LINESTYLES[label],
            linewidth=1.8,
        )
    ax.set_xlabel("Number of complete-placement evaluations")
    ax.set_ylabel("Best mean detection delay (h)")
    ax.set_xlim(1, queries[-1])
    ax.set_xticks([1, 50, 100, 150, 200])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False, ncol=2)
    save(fig, output, "best_so_far")


def plot_final_distribution(data, output):
    labels = list(data)
    finals = [data[label][:, -1] for label in labels]
    fig, ax = plt.subplots(figsize=(5.25, 3.25))
    parts = ax.violinplot(finals, showmeans=False, showmedians=False, showextrema=False)
    for body, label in zip(parts["bodies"], labels):
        body.set_facecolor(COLORS[label])
        body.set_edgecolor(COLORS[label])
        body.set_alpha(0.22)
    rng = np.random.default_rng(2026)
    for i, (label, values) in enumerate(zip(labels, finals), start=1):
        jitter = rng.uniform(-0.075, 0.075, size=len(values))
        ax.scatter(
            i + jitter, values, s=11, color=COLORS[label], alpha=0.65, linewidth=0
        )
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.vlines(i, q25, q75, color="black", linewidth=2.2)
        ax.scatter(i, median, marker="_", s=180, color="black", linewidth=1.6, zorder=4)
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=15, ha="right")
    ax.set_ylabel("Final best detection delay (h)")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    save(fig, output, "final_performance")


def plot_final_distribution_by_k(data_by_k, output):
    fig, axes = plt.subplots(1, len(data_by_k), figsize=(9.25, 2.85), sharex=True)
    short_labels = ["Random", "Local", "GraphComBO", "Proposed"]
    rng = np.random.default_rng(2026)
    for ax, (sensor_count, data) in zip(axes, data_by_k.items()):
        labels = list(data)
        finals = [data[label][:, -1] for label in labels]
        parts = ax.violinplot(
            finals, showmeans=False, showmedians=False, showextrema=False
        )
        for body, label in zip(parts["bodies"], labels):
            body.set_facecolor(COLORS[label])
            body.set_edgecolor(COLORS[label])
            body.set_alpha(0.22)
        for i, (label, values) in enumerate(zip(labels, finals), start=1):
            jitter = rng.uniform(-0.07, 0.07, size=len(values))
            ax.scatter(
                i + jitter,
                values,
                s=8,
                color=COLORS[label],
                alpha=0.60,
                linewidth=0,
            )
            q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
            ax.vlines(i, q25, q75, color="black", linewidth=1.8)
            ax.scatter(
                i,
                median,
                marker="_",
                s=125,
                color="black",
                linewidth=1.35,
                zorder=4,
            )
        ax.set_title(f"$K={sensor_count}$")
        ax.set_xticks(range(1, len(labels) + 1), short_labels, rotation=35, ha="right")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    axes[0].set_ylabel("Final best detection delay (h)")
    fig.subplots_adjust(wspace=0.24, bottom=0.28)
    save(fig, output, "final_performance_by_sensor_count")


def plot_graph_ablation(data, output):
    binary = data["GraphComBO"][:, -1]
    proposed = data["Proposed"][:, -1]
    fig, ax = plt.subplots(figsize=(3.65, 3.25))
    low = min(binary.min(), proposed.min()) - 0.15
    high = max(binary.max(), proposed.max()) + 0.15
    ax.plot([low, high], [low, high], color="#777777", linestyle="--", linewidth=1)
    improved = proposed < binary
    tied = proposed == binary
    colors = np.where(
        improved, COLORS["Proposed"], np.where(tied, "#7F7F7F", COLORS["GraphComBO"])
    )
    ax.scatter(
        binary, proposed, c=colors, s=25, alpha=0.8, edgecolor="white", linewidth=0.35
    )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("GraphComBO final delay (h)")
    ax.set_ylabel("Proposed final delay (h)")
    ax.text(
        0.04,
        0.96,
        f"Improved: {improved.sum()}\nTied: {tied.sum()}\nWorse: {(proposed > binary).sum()}",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
    )
    ax.grid(color="#E2E2E2", linewidth=0.5)
    save(fig, output, "graph_ablation")


def plot_best_so_far_by_k(data_by_k, output):
    fig, axes = plt.subplots(1, len(data_by_k), figsize=(9.25, 2.65), sharex=True)
    for ax, (sensor_count, data) in zip(axes, data_by_k.items()):
        queries = np.arange(1, next(iter(data.values())).shape[1] + 1)
        for label, values in data.items():
            median = np.median(values, axis=0)
            q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
            ax.fill_between(
                queries, q25, q75, color=COLORS[label], alpha=0.11, linewidth=0
            )
            ax.plot(
                queries,
                median,
                color=COLORS[label],
                linestyle=LINESTYLES[label],
                linewidth=1.45,
                label=label,
            )
        ax.set_title(f"$K={sensor_count}$")
        ax.set_xlim(1, queries[-1])
        ax.set_xticks([1, 100, 200])
        if sensor_count == 3:
            ax.set_ylim(4.0, 12.0)
        elif sensor_count == 5:
            ax.set_ylim(3.0, 10.0)
        elif sensor_count == 7:
            ax.set_ylim(2.0, 7.0)
        elif sensor_count == 9:
            ax.set_ylim(2.0, 6.0)
        else:
            raise NotImplementedError(f"Unexpected sensor count {sensor_count}")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        ax.set_xlabel("Evaluations")
    axes[0].set_ylabel("Best detection delay (h)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center")
    fig.subplots_adjust(top=0.78, wspace=0.25)
    save(fig, output, "best_so_far_by_sensor_count")


def plot_sensor_count_sensitivity(data_by_k, output):
    fig, ax = plt.subplots(figsize=(4.5, 3.15))
    sensor_counts = np.asarray(list(data_by_k))
    for label in next(iter(data_by_k.values())):
        finals = [data_by_k[k][label][:, -1] for k in sensor_counts]
        medians = np.asarray([np.median(values) for values in finals])
        q25 = np.asarray([np.quantile(values, 0.25) for values in finals])
        q75 = np.asarray([np.quantile(values, 0.75) for values in finals])
        ax.errorbar(
            sensor_counts,
            medians,
            yerr=[medians - q25, q75 - medians],
            label=label,
            color=COLORS[label],
            linestyle=LINESTYLES[label],
            marker="o",
            markersize=4,
            linewidth=1.6,
            capsize=2.5,
        )
    ax.set_xlabel("Number of sensors $K$")
    ax.set_ylabel("Final best detection delay (h)")
    ax.set_xticks(sensor_counts)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False, ncol=2)
    save(fig, output, "sensor_count_sensitivity")


def main():
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    artifact = Path(config["output_dir"])
    settings = config["search"]
    seeds = range(
        settings["first_seed"], settings["first_seed"] + settings["repetitions"]
    )
    budget = settings["budget"]

    def sources(sensor_count):
        suffix = "" if sensor_count == 5 else f"_k{sensor_count}"
        proposal_suffix = (
            "placement_coverage_kernel"
            if sensor_count == 5
            else f"placement_coverage_kernel_k{sensor_count}"
        )
        return {
            "Random search": (f"random_search{suffix}.csv", None),
            "Local search": (f"local_search{suffix}.csv", "local_search"),
            "GraphComBO": (
                f"original_graphcombo{suffix}.csv",
                "graphcombo_original_binary",
            ),
            "Proposed": (
                f"proposed_graphcombo_{proposal_suffix}.csv",
                "graphcombo_placement_coverage_kernel",
            ),
        }

    data_by_k = {
        sensor_count: {
            label: trajectories(load_rows(artifact / filename, method), seeds, budget)
            for label, (filename, method) in sources(sensor_count).items()
        }
        for sensor_count in (3, 5, 7, 9)
    }
    data = data_by_k[5]
    output = Path("figures")
    output.mkdir(exist_ok=True)
    style()
    plot_best_so_far(data, output)
    plot_final_distribution(data, output)
    plot_graph_ablation(data, output)
    plot_best_so_far_by_k(data_by_k, output)
    plot_final_distribution_by_k(data_by_k, output)
    plot_sensor_count_sensitivity(data_by_k, output)
    print("Wrote six publication figures to figures/ (.pdf/.png)")


if __name__ == "__main__":
    main()
