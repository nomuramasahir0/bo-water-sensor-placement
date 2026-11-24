# Expensive Water-Sensor Placement on Net3

This repository provides experiment code accompanying a paper on optimizing contamination-detection sensor placement in a water distribution network when only a limited number of complete placements can be evaluated.
The water-sensor code integrates with a separately obtained, pinned checkout of GraphComBO; GraphComBO source code is not redistributed here.
The experimental configuration is in [`config.yaml`](config.yaml).

The experiments use WNTR to run EPANET and generate contamination-propagation scenarios on Net3.
The following four methods are compared using identical initial placements and evaluation budgets:

- Uniform random search
- Single-sensor-swap local search
- GraphComBO using the binary pipe topology
- The proposed method using a diffusion kernel weighted by nearest-sensor travel-time coverage

The main experiments use sensor counts $K\in\{3,5,7,9\}$, 200 evaluations per run, and 30 repetitions with seeds 100--129.

## GraphComBO dependency and provenance

The binary-graph baseline uses the implementation from the [GraphComBO repository](https://github.com/LeonResearch/GraphComBO), associated with the [GraphComBO paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/437cd2749391ad40f67e4dd1d87c4596-Abstract-Conference.html).
Users must obtain that repository separately and check out commit `66a213434a344a84b9bfbbfddd680863422049b9`.
This repository imports its context construction, diffusion-kernel GP, acquisition, and trust-region components at runtime.
The water-sensor oracle, experiment pipeline, and hydraulic edge weighting are maintained here.

> **Third-party notice:** GraphComBO is not included in or licensed by this repository.
> Consult its authors and repository for the terms governing your use of that separately obtained code.

> **Reproducibility note:** The upstream commit and Python packages are pinned, but platform-specific numerical differences can still change Bayesian-optimization trajectories.

## 1. Environment Setup

Python 3.11 is recommended for compatibility with the pinned GraphComBO and BoTorch versions.
Clone GraphComBO under the ignored `external/` directory and expose its location:

```bash
git clone https://github.com/LeonResearch/GraphComBO.git external/GraphComBO
git -C external/GraphComBO checkout 66a213434a344a84b9bfbbfddd680863422049b9
export GRAPHCOMBO_ROOT="$PWD/external/GraphComBO"
```

Then create the environment from the repository root:

```bash
python3.11 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt
```

Verify the external checkout and Python dependencies, then run the repository tests:

```bash
venv/bin/python -c "from original_graphcombo import load_graphcombo; load_graphcombo(); print('GraphComBO ready')"
venv/bin/python -m pytest -q
```

## 2. Generate Contamination Scenarios and the Evaluation Oracle

```bash
venv/bin/python prepare_delays.py
```

This script uses the EPANET Net3 model included with WNTR to:

- Treat all 92 junctions as candidate sensor locations
- Select 79 propagation-capable sources using hydraulic information only
- Combine the injection start times `0, 6, 12, 18, 24 h` to create 395 scenarios
- Construct a scenario-by-candidate-junction detection-delay matrix from a 48-hour water-quality simulation
- Assign a 48-hour penalty to source-sensor pairs for which contamination is not detected within 48 hours

The main outputs are:

```text
artifacts/water_sensor/detection_delays.npz
artifacts/water_sensor/detection_delays.json
```

`detection_delays.npz` is the oracle shared by all optimization methods.
Its generation requires 395 EPANET simulations.
The following command can be used for a quick test, but its output is not suitable for the main experiments:

```bash
venv/bin/python prepare_delays.py --limit-sources 1
```

After this test, rerun `prepare_delays.py` without any options before conducting the main experiments.

## 3. Experiments Using the Paper Settings

The following commands overwrite existing CSV files with the same names.
First, run the primary setting, $K=5$:

```bash
venv/bin/python random_search.py
venv/bin/python run_local_search.py
venv/bin/python run_original_graphcombo.py
venv/bin/python run_proposed_graphcombo.py \
  --output-suffix placement_coverage_kernel
```

For $K=3,7,9$, use the suffixes expected by the aggregation scripts:

```bash
for k in 3 7 9; do
  venv/bin/python random_search.py \
    --sensor-count "$k" --output-suffix "k$k"
  venv/bin/python run_local_search.py \
    --sensor-count "$k" --output-suffix "k$k"
  venv/bin/python run_original_graphcombo.py \
    --sensor-count "$k" --output-suffix "k$k"
  venv/bin/python run_proposed_graphcombo.py \
    --sensor-count "$k" --output-suffix "placement_coverage_kernel_k$k"
done
```

Under the main settings in `config.yaml`, each command performs 200 evaluations for each of seeds 100--129.
For a given seed, all methods use the same 10 initial random placements.
Local search and both BO methods use the same global-restart rule: restart after 20 evaluations without a relative improvement of at least `1e-3`.

The following is an example of a short smoke test.
A suffix is specified to avoid overwriting the main experimental CSV files:

```bash
venv/bin/python run_proposed_graphcombo.py \
  --seeds 1 --first-seed 0 --budget 20 --output-suffix smoke
```

## 4. Generate Summary Statistics and Figures

After all methods have completed for all four values of $K$, generate the summary statistics and figures:

```bash
venv/bin/python summarize_final_experiment.py
venv/bin/python plot_experiment_results.py
```

The summary is written to `artifacts/water_sensor/final_experiment_summary.json`.
Figures are written in PDF and PNG formats to `figures/`.
The main figures are:

- `best_so_far.pdf`: median best-so-far trajectory and IQR for $K=5$
- `final_performance.pdf`: distribution of final values for $K=5$
- `best_so_far_by_sensor_count.pdf`: best-so-far trajectories for $K=3,5,7,9$
- `sensor_count_sensitivity.pdf`: final performance by sensor count

Regenerate the Net3/WNTR/EPAN overview figure with:

```bash
venv/bin/python plot_problem_overview.py
```

This script reads the best $K=5$ placement found during the experiments from the main CSV file for the proposed method.

## 5. Hyperparameter Sensitivity Analysis

The proposed kernel uses `alpha=4` and the dimensionless bandwidth `h=1` in the paper.
Run the one-factor-at-a-time sensitivity analysis for $K=5$ and seeds 100--129 with:

```bash
venv/bin/python run_hyperparameter_sensitivity.py --jobs 3
venv/bin/python plot_hyperparameter_sensitivity.py
```

`--jobs` controls the number of concurrent jobs.
Use `--jobs 1` if memory is limited.
Completed settings are skipped based on the summary JSON.
Note that `h=1` was selected after examining this sensitivity analysis; consequently, the 30 paper seeds are not an independent post-tuning holdout set.

## Implementation and Configuration Map

- `prepare_delays.py`: WNTR/EPANET scenarios and detection-delay matrix
- `random_search.py`: scalar complete-placement oracle and random search
- `run_local_search.py`: local-search baseline
- `run_original_graphcombo.py`: water-sensor runner using external GraphComBO
- `run_proposed_graphcombo.py`: proposed travel-time coverage kernel
- `original_graphcombo.py`: external-dependency loader and water-sensor integration
- `config.yaml`: simulation, search, and kernel settings used in the paper

The proposed method represents each placement $S$ by the dimensionless travel time from every junction to its nearest sensor.
It estimates the median feature distance from 20,000 objective-independent legal swaps and uses this scale to normalize the RBF distance.
The proposed method changes only the conductance and diffusion kernel on the combinatorial graph.
The binary and proposed methods use the external GraphComBO implementation for the legal-swap topology, diffusion-kernel Gaussian process, acquisition rule, context construction, and trust-region state.
This repository supplies the common initial placements, evaluation budget and restart policy, plus the hydraulic conductance used by the proposed method.
