"""Summarize the fixed-configuration seed 100--129 experiment."""

import csv
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import wilcoxon


def load(path, method=None):
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if method is not None:
        rows = [row for row in rows if row.get("method") == method]
    return {(int(row["seed"]), int(row["query"])): row for row in rows}


def best_so_far(rows, seed, budget):
    objectives = np.asarray([
        float(rows[seed, query]["objective_hours"])
        for query in range(1, budget + 1)
    ])
    return np.minimum.accumulate(objectives)


def main():
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    out = Path(config["output_dir"])
    settings = config["search"]
    seeds = list(range(settings["first_seed"], settings["first_seed"] + settings["repetitions"]))
    budget = settings["budget"]
    methods = {
        "random_search": load(out / "random_search.csv"),
        "local_search": load(out / "local_search.csv", "local_search"),
        "graphcombo_binary": load(out / "original_graphcombo.csv"),
        "placement_coverage_kernel": load(
            out / "proposed_graphcombo_placement_coverage_kernel.csv"
        ),
    }
    summary = {"seeds": seeds, "budget": budget, "methods": {}}
    finals = {}
    for name, rows in methods.items():
        traces = np.asarray([best_so_far(rows, seed, budget) for seed in seeds])
        final = traces[:, -1]
        finals[name] = final
        auc = traces.sum(axis=1)
        checkpoints = {}
        for query in (10, 20, 50, 100, 200):
            if query > budget:
                continue
            values = traces[:, query - 1]
            checkpoints[str(query)] = {
                "median_hours": float(np.median(values)),
                "q25_hours": float(np.quantile(values, 0.25)),
                "q75_hours": float(np.quantile(values, 0.75)),
            }
        summary["methods"][name] = {
            "checkpoints": checkpoints,
            "median_auc": float(np.median(auc)),
        }
    proposed = finals["placement_coverage_kernel"]
    binary = finals["graphcombo_binary"]
    summary["paired_proposal_vs_binary"] = {
        "wins": int(np.sum(proposed < binary)),
        "ties": int(np.sum(proposed == binary)),
        "losses": int(np.sum(proposed > binary)),
        "mean_difference_hours": float(np.mean(proposed - binary)),
        "wilcoxon_one_sided_p": float(wilcoxon(proposed, binary, alternative="less").pvalue),
    }
    path = out / "final_experiment_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
