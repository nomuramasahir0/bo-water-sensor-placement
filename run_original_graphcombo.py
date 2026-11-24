"""Run the binary-graph baseline with a separately installed GraphComBO checkout."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from original_graphcombo import binary_base_graph, run_graph_bayesian_optimization
from random_search import DetectionDelayOracle
from search_methods import initial_placements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--matrix", default=None)
    parser.add_argument("--seeds", type=int, default=None, help="Override repetitions")
    parser.add_argument("--first-seed", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--sensor-count", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument(
        "--graphcombo-root",
        default=None,
        help="Path to the pinned external GraphComBO checkout (or set GRAPHCOMBO_ROOT)",
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(config["output_dir"])
    matrix = Path(args.matrix) if args.matrix else out_dir / "detection_delays.npz"
    oracle = DetectionDelayOracle(matrix)
    base_graph = binary_base_graph(matrix)
    common = config["search"]
    repetitions = args.seeds or common["repetitions"]
    budget = args.budget or common["budget"]
    sensor_count = args.sensor_count or config["sensor_count"]
    rows = []
    first_seed = common["first_seed"] if args.first_seed is None else args.first_seed
    for seed in range(first_seed, first_seed + repetitions):
        print(f"seed={seed} method=graphcombo_original_binary")
        initial = initial_placements(
            len(oracle), sensor_count, common["initial_evaluations"], seed
        )
        rows.extend(
            run_graph_bayesian_optimization(
                oracle, base_graph, budget, seed, initial,
                config["original_graphcombo"],
                graphcombo_root=args.graphcombo_root,
            )
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    path = out_dir / f"original_graphcombo{suffix}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    final = [r["best_so_far_hours"] for r in rows if r["query"] == budget]
    summary = {
        "median_final_best_hours": float(np.median(final)),
        "q25_final_best_hours": float(np.quantile(final, 0.25)),
        "q75_final_best_hours": float(np.quantile(final, 0.75)),
        "repetitions": repetitions,
        "budget": budget,
        "sensor_count": sensor_count,
    }
    (out_dir / f"original_graphcombo_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
