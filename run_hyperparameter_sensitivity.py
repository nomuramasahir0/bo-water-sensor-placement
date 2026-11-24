"""Run the one-at-a-time sensitivity study for the proposed kernel."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULTS = {"alpha": 4.0, "bandwidth": 1.0}
GRIDS = {
    "alpha": [1.0, 2.0, 8.0],  # alpha=0 and 4 reuse formal experiments
    "bandwidth": [0.5, 1.5, 2.0],
}


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def expected_summary(output_dir: Path, parameter: str, value: float) -> Path:
    return output_dir / f"proposed_graphcombo_summary_travel_sensitivity_{parameter}_{tag(value)}.json"


def is_complete(path: Path, seeds: int, first_seed: int, budget: int) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    return (
        data.get("repetitions") == seeds
        and data.get("budget") == budget
        and data.get("sensor_count") == 5
        and data.get("first_seed", 100) == first_seed
    )


def command(parameter: str, value: float, seeds: int, first_seed: int, budget: int):
    values = DEFAULTS | {parameter: value}
    return [
        sys.executable,
        "run_proposed_graphcombo.py",
        "--sensor-count", "5",
        "--seeds", str(seeds),
        "--first-seed", str(first_seed),
        "--budget", str(budget),
        "--kernel-alpha", str(values["alpha"]),
        "--kernel-bandwidth", str(values["bandwidth"]),
        "--output-suffix", f"travel_sensitivity_{parameter}_{tag(value)}",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--first-seed", type=int, default=100)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--output-dir", default="artifacts/water_sensor")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    output_dir = Path(args.output_dir)
    tasks = []
    for parameter, values in GRIDS.items():
        for value in values:
            summary = expected_summary(output_dir, parameter, value)
            if is_complete(summary, args.seeds, args.first_seed, args.budget):
                print(f"skip complete: {parameter}={value:g}", flush=True)
            else:
                tasks.append((parameter, value))

    def run(task):
        parameter, value = task
        log = output_dir / f"travel_sensitivity_{parameter}_{tag(value)}.log"
        print(f"start: {parameter}={value:g} (log: {log})", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(
                command(parameter, value, args.seeds, args.first_seed, args.budget),
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        print(f"complete: {parameter}={value:g}", flush=True)
        return task

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run, task) for task in tasks]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
