"""Run only the local-search baseline on the fixed-cardinality oracle."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from random_search import DetectionDelayOracle
from search_methods import CandidateGraph, initial_placements, run_local_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--matrix", default=None)
    parser.add_argument("--seeds", type=int, default=None, help="Override repetitions")
    parser.add_argument("--first-seed", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--sensor-count", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(config["output_dir"])
    matrix = Path(args.matrix) if args.matrix else out_dir / "detection_delays.npz"
    oracle = DetectionDelayOracle(matrix)
    settings = config["search"]
    graph = CandidateGraph.from_matrix(matrix)
    repetitions = args.seeds or settings["repetitions"]
    first_seed = settings["first_seed"] if args.first_seed is None else args.first_seed
    budget = args.budget or settings["budget"]
    sensor_count = args.sensor_count or config["sensor_count"]
    rows = []
    for seed in range(first_seed, first_seed + repetitions):
        print(f"seed={seed} method=local_search")
        initial = initial_placements(
            len(oracle), sensor_count, settings["initial_evaluations"], seed
        )
        rows.extend(run_local_search(
            oracle, graph, sensor_count, budget, seed, initial, settings
        ))
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    with (out_dir / f"local_search{suffix}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    final = [row["best_so_far_hours"] for row in rows if row["query"] == budget]
    summary = {
        "median_final_best_hours": float(np.median(final)),
        "q25_final_best_hours": float(np.quantile(final, 0.25)),
        "q75_final_best_hours": float(np.quantile(final, 0.75)),
        "repetitions": repetitions,
        "budget": budget,
        "sensor_count": sensor_count,
    }
    (out_dir / f"local_search_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
