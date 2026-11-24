"""Complete-placement oracle and uniform random-search baseline."""

from __future__ import annotations

import argparse
import csv
import json
from math import comb
from pathlib import Path

import numpy as np
import yaml


class DetectionDelayOracle:
    """Expose only scalar placement evaluations from a delay matrix."""

    def __init__(self, matrix_file: str | Path):
        data = np.load(matrix_file, allow_pickle=False)
        self.delays = np.asarray(data["delays_hours"], dtype=float)
        self.candidate_names = np.asarray(data["candidate_names"]).astype(str)
        if self.delays.ndim != 2 or self.delays.shape[1] != len(self.candidate_names):
            raise ValueError("Invalid delay matrix: expected scenarios x candidates")

    def __len__(self) -> int:
        return len(self.candidate_names)

    def evaluate(self, placement: np.ndarray) -> float:
        placement = np.asarray(placement, dtype=int)
        if placement.ndim != 1 or len(np.unique(placement)) != len(placement):
            raise ValueError("A placement must contain distinct candidate indices")
        if np.any(placement < 0) or np.any(placement >= len(self)):
            raise IndexError("Placement contains an invalid candidate index")
        return float(np.min(self.delays[:, placement], axis=1).mean())


def run_random_search(
    oracle: DetectionDelayOracle, sensor_count: int, budget: int, seed: int
) -> list[dict]:
    if not 1 <= sensor_count <= len(oracle):
        raise ValueError("sensor_count must be between 1 and the candidate count")
    if budget < 1:
        raise ValueError("budget must be positive")
    search_space_size = comb(len(oracle), sensor_count)
    if budget > search_space_size:
        raise ValueError(
            f"budget={budget} exceeds the {search_space_size} distinct placements"
        )

    rng = np.random.default_rng(seed)
    seen: set[tuple[int, ...]] = set()
    rows: list[dict] = []
    best_value = float("inf")
    best_placement: tuple[int, ...] | None = None

    while len(rows) < budget:
        placement = tuple(sorted(rng.choice(len(oracle), sensor_count, replace=False)))
        if placement in seen:
            continue
        seen.add(placement)
        value = oracle.evaluate(np.asarray(placement))
        if value < best_value:
            best_value = value
            best_placement = placement
        rows.append(
            {
                "seed": seed,
                "query": len(rows) + 1,
                "objective_hours": value,
                "best_so_far_hours": best_value,
                "placement": "|".join(oracle.candidate_names[list(placement)]),
                "best_placement": "|".join(
                    oracle.candidate_names[list(best_placement)]
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--matrix", default=None)
    parser.add_argument("--sensor-count", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--first-seed", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    out_dir = Path(config["output_dir"])
    matrix_file = Path(args.matrix) if args.matrix else out_dir / "detection_delays.npz"
    oracle = DetectionDelayOracle(matrix_file)
    settings = config["random_search"]
    sensor_count = args.sensor_count or config["sensor_count"]
    repetitions = args.seeds or settings["repetitions"]
    first_seed = settings["first_seed"] if args.first_seed is None else args.first_seed
    budget = args.budget or settings["budget"]

    all_rows: list[dict] = []
    for seed in range(first_seed, first_seed + repetitions):
        all_rows.extend(
            run_random_search(oracle, sensor_count, budget, seed)
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    result_file = out_dir / f"random_search{suffix}.csv"
    with result_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    final = [row["best_so_far_hours"] for row in all_rows if row["query"] == budget]
    summary = {
        "matrix": str(matrix_file),
        "candidate_count": len(oracle),
        "sensor_count": sensor_count,
        "budget": budget,
        "repetitions": repetitions,
        "median_final_best_hours": float(np.median(final)),
        "q25_final_best_hours": float(np.quantile(final, 0.25)),
        "q75_final_best_hours": float(np.quantile(final, 0.75)),
    }
    (out_dir / f"random_search_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
