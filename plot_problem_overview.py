"""Plot a paper-ready overview of the EPANET/WNTR sensor-placement problem."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from tempfile import gettempdir

import numpy as np
import yaml

# Keep plotting caches out of the user's home directory on shared machines.
cache_root = Path(gettempdir()) / "graphcombo-water-sensor-cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from prepare_delays import _net3_path


def best_placement(path: Path) -> tuple[list[str], float]:
    """Return the best placement encountered in an experiment CSV."""
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    row = min(rows, key=lambda item: float(item["best_so_far_hours"]))
    return row["best_placement"].split("|"), float(row["best_so_far_hours"])


def illustrative_scenario(
    delays: np.ndarray,
    sources: np.ndarray,
    starts: np.ndarray,
    candidate_names: np.ndarray,
    placement: list[str],
) -> int:
    """Select a reproducible, moderately spreading scenario for illustration."""
    lookup = {name: index for index, name in enumerate(candidate_names)}
    sensor_columns = [lookup[name] for name in placement]
    earliest_sensor_delay = delays[:, sensor_columns].min(axis=1)
    detected_fraction = (delays < 48.0).mean(axis=1)
    # A roughly three-hour detection and 60% network reach make propagation visible.
    score = np.abs(earliest_sensor_delay - 3.0) + 2.0 * np.abs(detected_fraction - 0.6)
    score += np.asarray([10.0 if source in placement else 0.0 for source in sources])
    return int(np.argmin(score))


def add_box(ax, xy, width, height, title, body, color):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=color,
        edgecolor="#5B6573",
        linewidth=0.8,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + 0.035,
        xy[1] + height - 0.04,
        title,
        transform=ax.transAxes,
        fontsize=8.8,
        fontweight="bold",
        va="top",
    )
    ax.text(
        xy[0] + 0.035,
        xy[1] + height - 0.12,
        body,
        transform=ax.transAxes,
        fontsize=8.7,
        linespacing=1.22,
        va="top",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--results",
        default="artifacts/water_sensor/proposed_graphcombo_placement_coverage_kernel.csv",
    )
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    import wntr

    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    network = wntr.network.WaterNetworkModel(
        str(_net3_path(wntr, config.get("network_file")))
    )
    archive = np.load(Path(config["output_dir"]) / "detection_delays.npz")
    names = archive["candidate_names"].astype(str)
    delays = archive["delays_hours"]
    sources = archive["scenario_sources"].astype(str)
    starts = archive["scenario_start_hours"]
    placement, objective = best_placement(Path(args.results))
    scenario = illustrative_scenario(delays, sources, starts, names, placement)

    coordinates = {
        name: np.asarray(network.get_node(name).coordinates, dtype=float)
        for name in network.node_name_list
    }
    segments = [
        [
            coordinates[network.get_link(name).start_node_name],
            coordinates[network.get_link(name).end_node_name],
        ]
        for name in network.link_name_list
    ]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0,
        }
    )
    fig = plt.figure(figsize=(7.15, 3.50))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0)
    ax = fig.add_subplot(grid[0, 0])
    info = fig.add_subplot(grid[0, 1])

    ax.add_collection(
        LineCollection(segments, colors="#AEB7C2", linewidths=1.0, zorder=1)
    )
    candidate_xy = np.asarray([coordinates[name] for name in names])
    arrival = delays[scenario]
    detected = arrival < 48.0
    ax.scatter(
        candidate_xy[~detected, 0],
        candidate_xy[~detected, 1],
        s=38,
        color="#AEB7C2",
        edgecolor="white",
        linewidth=0.25,
        zorder=2,
    )
    points = ax.scatter(
        candidate_xy[detected, 0],
        candidate_xy[detected, 1],
        c=arrival[detected],
        cmap="YlOrRd",
        vmin=0,
        vmax=max(6.0, float(np.quantile(arrival[detected], 0.9))),
        s=40,
        edgecolor="white",
        linewidth=0.25,
        zorder=3,
    )

    sensor_xy = np.asarray([coordinates[name] for name in placement])
    ax.scatter(
        sensor_xy[:, 0],
        sensor_xy[:, 1],
        marker="^",
        s=122,
        facecolor="#0072B2",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    source = sources[scenario]
    source_xy = coordinates[source]
    ax.scatter(
        source_xy[0],
        source_xy[1],
        marker="*",
        s=140,
        facecolor="#D55E00",
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
    )
    sensor_columns = [int(np.flatnonzero(names == name)[0]) for name in placement]
    first = sensor_columns[int(np.argmin(arrival[sensor_columns]))]
    first_xy = candidate_xy[first]
    ax.scatter(
        first_xy[0],
        first_xy[1],
        marker="o",
        s=155,
        facecolor="none",
        edgecolor="#003B5C",
        linewidth=1.3,
        zorder=4,
    )

    # ax.set_title("Illustrative contamination scenario on EPANET Net3", pad=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.04)
    ax.axis("off")
    colorbar = fig.colorbar(
        points, ax=ax, orientation="horizontal", fraction=0.050, pad=0.045
    )
    colorbar.set_label("Contaminant arrival time after injection (h)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7.5, length=2)
    legend = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor="#AEB7C2",
            markeredgecolor="white",
            label="Not reached by the end",
        ),
        Line2D(
            [],
            [],
            marker="*",
            linestyle="",
            markersize=14,
            markerfacecolor="#D55E00",
            markeredgecolor="white",
            label="Injection source",
        ),
        Line2D(
            [],
            [],
            marker="^",
            linestyle="",
            markersize=10,
            markerfacecolor="#0072B2",
            markeredgecolor="white",
            label="Selected sensor",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=9,
            markerfacecolor="none",
            markeredgecolor="#003B5C",
            label="First detecting sensor",
        ),
    ]
    ax.legend(
        handles=legend,
        loc="lower left",
        frameon=False,
        fontsize=8.7,
        handletextpad=0.3,
        borderaxespad=0.1,
    )

    info.set_xlim(0, 1)
    info.set_ylim(0, 1)
    info.axis("off")
    add_box(
        info,
        (0.02, 0.68),
        0.90,
        0.22,
        "1. Define a scenario with WNTR",
        "Network, demands, source node,\ninjection time, and duration",
        "#EDF5FA",
    )
    add_box(
        info,
        (0.02, 0.37),
        0.90,
        0.22,
        "2. Simulate with EPANET",
        "Solve hydraulics and water quality;\nrecord delay $D_{\\omega v}$ at every junction",
        "#FFF5E8",
    )
    add_box(
        info,
        (0.02, 0.035),
        0.90,
        0.25,
        "3. Optimize a sensor subset",
        "Choose $|S|=K$ junctions as sensors\nMinimize the delay over all scenarios",
        "#EAF6F1",
    )
    for y0, y1 in [(0.68, 0.61), (0.37, 0.30)]:
        info.add_patch(
            FancyArrowPatch(
                (0.50, y0),
                (0.50, y1),
                transform=info.transAxes,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color="#5B6573",
            )
        )
    info.text(
        0.50,
        0.98,
        "Simulation workflow",
        transform=info.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    detected_delay = float(arrival[first])
    fig.subplots_adjust(bottom=0.08, top=0.94, left=0, right=1)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(output / f"problem_overview.{extension}", **kwargs)
    plt.close(fig)
    print(
        f"Wrote {output / 'problem_overview.pdf'} and PNG "
        f"(source={source}, start={starts[scenario]:g} h, delay={detected_delay:g} h)"
    )


if __name__ == "__main__":
    main()
