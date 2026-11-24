"""Plot paired final-performance sensitivity of the proposed kernel."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARTIFACTS = Path("artifacts/water_sensor")
FIGURES = Path("figures")
DEFAULT_FILE = ARTIFACTS / "proposed_graphcombo_placement_coverage_kernel.csv"
ORIGINAL_FILE = ARTIFACTS / "original_graphcombo.csv"
DEFAULTS = {"alpha": 4.0, "bandwidth": 1.0}
GRIDS = {
    "alpha": [0.0, 1.0, 2.0, 4.0, 8.0],
    "bandwidth": [0.5, 1.0, 1.5, 2.0],
}
LABELS = {
    "alpha": r"Edge strength $\alpha$",
    "bandwidth": r"Bandwidth $h$",
}


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def final_by_seed(path: Path) -> pd.Series:
    data = pd.read_csv(path)
    return data.groupby("seed")["objective_hours"].min().sort_index()


def file_for(parameter: str, value: float) -> Path:
    if parameter == "alpha" and value == 0:
        return ORIGINAL_FILE
    if value == DEFAULTS[parameter]:
        return DEFAULT_FILE
    return ARTIFACTS / f"proposed_graphcombo_travel_sensitivity_{parameter}_{tag(value)}.csv"


def main():
    baseline = final_by_seed(DEFAULT_FILE)
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.0), sharey=True)
    for axis, (parameter, values) in zip(axes, GRIDS.items()):
        means, standard_errors = [], []
        for value in values:
            path = file_for(parameter, value)
            if not path.exists():
                raise FileNotFoundError(f"Run sensitivity experiment first: {path}")
            variant = final_by_seed(path)
            common = baseline.index.intersection(variant.index)
            delta = (variant.loc[common] - baseline.loc[common]).to_numpy()
            final = variant.loc[common].to_numpy()
            mean = float(np.mean(final))
            standard_error = float(np.std(final, ddof=1) / np.sqrt(len(final)))
            means.append(mean)
            standard_errors.append(standard_error)
            records.append(
                {
                    "parameter": parameter,
                    "value": value,
                    "seeds": len(delta),
                    "mean_final_hours": mean,
                    "standard_error_hours": standard_error,
                    "median_final_hours": float(np.median(final)),
                    "median_paired_delta_hours": float(np.median(delta)),
                    "wins_vs_default": int(np.sum(delta < 0)),
                    "ties_vs_default": int(np.sum(delta == 0)),
                }
            )
        x = np.arange(len(values))
        axis.errorbar(
            x, means, yerr=standard_errors, fmt="o-", capsize=3, color="#0072B2"
        )
        default_index = values.index(DEFAULTS[parameter])
        axis.scatter(
            x[default_index],
            means[default_index],
            marker="*",
            s=90,
            color="#D55E00",
            zorder=3,
        )
        axis.set_xticks(x, [f"{value:g}" for value in values])
        axis.set_xlabel(LABELS[parameter])
        axis.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Final best value (h)")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "hyperparameter_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(
        FIGURES / "hyperparameter_sensitivity.png", dpi=300, bbox_inches="tight"
    )
    table = pd.DataFrame(records)
    table.to_csv(ARTIFACTS / "hyperparameter_sensitivity.csv", index=False)
    summary = {
        parameter: table[table.parameter == parameter]
        .drop(columns="parameter")
        .to_dict("records")
        for parameter in GRIDS
    }
    (ARTIFACTS / "hyperparameter_sensitivity.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
