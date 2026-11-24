"""Draw the conceptual pipeline of the hydraulic coverage kernel."""

from __future__ import annotations

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize

FIGURES = Path("figures")


def network():
    positions = {
        0: (0.0, 2.0),
        1: (1.0, 2.0),
        2: (2.0, 2.0),
        3: (0.0, 1.0),
        4: (1.0, 1.0),
        5: (2.0, 1.0),
        6: (0.0, 0.0),
        7: (1.0, 0.0),
        8: (2.0, 0.0),
    }
    # Illustrative representative pipe traversal times (dimensionless here).
    edges = {
        (0, 1): 1.0,
        (1, 2): 2.8,
        (0, 3): 1.8,
        (1, 4): 0.8,
        (2, 5): 1.1,
        (3, 4): 1.2,
        (4, 5): 2.5,
        (3, 6): 2.7,
        (4, 7): 0.9,
        (5, 8): 1.4,
        (6, 7): 1.0,
        (7, 8): 2.2,
    }
    graph = nx.Graph()
    graph.add_nodes_from(positions)
    graph.add_weighted_edges_from((u, v, w) for (u, v), w in edges.items())
    return graph, positions, edges


def coverage(graph, placement):
    scale = float(np.median([data["weight"] for _, _, data in graph.edges(data=True)]))
    return (
        np.asarray(
            [
                min(
                    nx.shortest_path_length(graph, j, sensor, weight="weight")
                    for sensor in placement
                )
                for j in graph.nodes
            ]
        )
        / scale
    )


def draw_network(ax, graph, positions, values=None, placement=(), labels=False):
    edge_color = mpl.rcParams["axes.edgecolor"]
    nx.draw_networkx_edges(
        graph, positions, ax=ax, width=1.25, edge_color=edge_color, alpha=0.72
    )
    if values is None:
        colors = [mpl.rcParams["axes.facecolor"]] * len(graph)
    else:
        colors = values
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_size=165,
        node_color=colors,
        cmap=mpl.colormaps["Blues_r"],
        vmin=0,
        vmax=3.2,
        edgecolors=edge_color,
        linewidths=0.8,
    )
    if placement:
        xs = [positions[v][0] for v in placement]
        ys = [positions[v][1] for v in placement]
        ax.scatter(
            xs,
            ys,
            marker="*",
            s=105,
            color="#d95f02",
            edgecolor="white",
            linewidth=0.55,
            zorder=5,
        )
    if labels:
        nx.draw_networkx_labels(graph, positions, ax=ax, font_size=7)
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-0.3, 2.3)
    ax.axis("off")


def main():
    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    graph, positions, edge_times = network()
    fig = plt.figure(figsize=(9.2, 2.45), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.9, 3.4],
        left=0.025,
        right=0.99,
        bottom=0.13,
        top=0.84,
        wspace=0.050,
    )

    # (a) Static hydraulic proxy on the physical junction graph.
    ax_a = fig.add_subplot(grid[0, 0])
    ax_a.set_title("(a) Hydraulic distance", loc="center", pad=8)
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax_a,
        width=[3.1 / np.sqrt(edge_times[e]) for e in graph.edges],
        edge_color=[edge_times[e] for e in graph.edges],
        edge_cmap=mpl.colormaps["cividis"],
        edge_vmin=0.7,
        edge_vmax=2.9,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax_a,
        node_size=100,
        node_color="white",
        edgecolors="0.2",
        linewidths=0.8,
    )
    ax_a.text(
        1.0,
        -0.48,
        r"edge cost $\bar{\tau}_e=L_e/(|q_e|/a_e)$",
        ha="center",
        va="top",
        fontsize=9.3,
    )
    ax_a.text(
        1.0,
        -0.78,
        "shortest paths define $d_\\tau(u,v)$",
        ha="center",
        va="top",
        fontsize=9.3,
    )
    ax_a.set_aspect("equal")
    ax_a.set_xlim(-0.3, 2.3)
    ax_a.set_ylim(-0.85, 2.3)
    ax_a.axis("off")

    # (b) Contrast one coverage-preserving swap with two disruptive swaps.
    ax_b = fig.add_subplot(grid[0, 1])
    ax_b.set_title(
        "(b) Coverage similarity between sensor placements", loc="center", pad=8
    )
    ax_b.axis("off")
    box = ax_b.get_position()
    x_offsets = {"S": 0.00, "Sp": 0.255, "S2": 0.510, "S3": 0.765}
    mini = {
        key: fig.add_axes(
            [
                box.x0 + box.width * offset,
                box.y0 + box.height * 0.20,
                box.width * 0.215,
                box.height * 0.66,
            ]
        )
        for key, offset in x_offsets.items()
    }
    placement_s = (0, 6, 8)
    placement_sp = (0, 7, 8)  # legal one-pipe move 6 -> 7
    placement_s2 = (0, 3, 8)  # legal one-pipe move 6 -> 3
    placement_s3 = (0, 6, 7)  # legal one-pipe move 8 -> 7
    r_s = coverage(graph, placement_s)
    r_sp = coverage(graph, placement_sp)
    r_s2 = coverage(graph, placement_s2)
    r_s3 = coverage(graph, placement_s3)
    specs = [
        ("S", placement_s, r_s, r"$S=\{0,6,8\}$"),
        ("Sp", placement_sp, r_sp, r"$S'=\{0,7,8\}$"),
        ("S2", placement_s2, r_s2, r"$S_2=\{0,3,8\}$"),
        ("S3", placement_s3, r_s3, r"$S_3=\{0,6,7\}$"),
    ]
    for key, placement, values, title in specs:
        draw_network(mini[key], graph, positions, values, placement)
        mini[key].set_title(title, fontsize=10, fontweight="normal", pad=0)

    distances = {
        "Sp": float(np.linalg.norm(r_s - r_sp)),
        "S2": float(np.linalg.norm(r_s - r_s2)),
        "S3": float(np.linalg.norm(r_s - r_s3)),
    }
    centers = {key: offset + 0.1075 for key, offset in x_offsets.items()}
    ax_b.text(
        centers["S"],
        0.06,
        "reference",
        transform=ax_b.transAxes,
        ha="center",
        va="center",
        fontsize=9.3,
    )
    ax_b.text(
        centers["Sp"],
        0.06,
        rf"$\|r_S-r_{{S'}}\|_2={distances['Sp']:.1f}$" + "\nhigh similarity",
        transform=ax_b.transAxes,
        ha="center",
        va="center",
        fontsize=9.3,
        color="#18794e",
    )
    ax_b.text(
        centers["S2"],
        0.06,
        rf"$\|r_S-r_{{S_2}}\|_2={distances['S2']:.1f}$" + "\nlow similarity",
        transform=ax_b.transAxes,
        ha="center",
        va="center",
        fontsize=9.3,
        color="#b42318",
    )
    ax_b.text(
        centers["S3"],
        0.06,
        rf"$\|r_S-r_{{S_3}}\|_2={distances['S3']:.1f}$" + "\nlow similarity",
        transform=ax_b.transAxes,
        ha="center",
        va="center",
        fontsize=9.3,
        color="#b42318",
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(
            FIGURES / f"hydraulic_coverage_kernel.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
