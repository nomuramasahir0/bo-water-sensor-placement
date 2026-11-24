"""Water-sensor integration for a separately installed GraphComBO checkout.

GraphComBO is not distributed in this repository. Set ``GRAPHCOMBO_ROOT`` or
pass ``graphcombo_root`` to point at the pinned upstream checkout documented in
the README.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import random
import subprocess
import sys

import networkx as nx
import numpy as np

from random_search import DetectionDelayOracle
from search_methods import _record

GRAPHCOMBO_COMMIT = "66a213434a344a84b9bfbbfddd680863422049b9"
GRAPHCOMBO_URL = "https://github.com/LeonResearch/GraphComBO.git"


class GraphComBODependencyError(RuntimeError):
    """Raised when the separately supplied GraphComBO checkout is unusable."""


@dataclass(frozen=True)
class _GraphComBOAPI:
    torch: object
    standardize: object
    initialize_model: object
    get_acqf: object
    construct_context: object
    TrustRegionState: object
    update_state: object


def _setup_message(reason: str) -> str:
    return (
        f"{reason}\n\n"
        "Prepare the external dependency with:\n"
        f"  git clone {GRAPHCOMBO_URL} /path/to/GraphComBO\n"
        f"  git -C /path/to/GraphComBO checkout {GRAPHCOMBO_COMMIT}\n"
        "  export GRAPHCOMBO_ROOT=/path/to/GraphComBO\n"
        "Then install this repository's requirements in the same environment."
    )


def load_graphcombo(graphcombo_root: str | Path | None = None) -> _GraphComBOAPI:
    """Validate and import the pinned external GraphComBO checkout."""
    configured = graphcombo_root or os.environ.get("GRAPHCOMBO_ROOT")
    if not configured:
        raise GraphComBODependencyError(
            _setup_message("GraphComBO is required but GRAPHCOMBO_ROOT is not set.")
        )
    root = Path(configured).expanduser().resolve()
    required = [
        root / "search" / "models.py",
        root / "search" / "kernels.py",
        root / "search" / "self_combograph.py",
        root / "search" / "trust_region.py",
        root / "search" / "utils.py",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise GraphComBODependencyError(
            _setup_message(
                f"GRAPHCOMBO_ROOT={root} is missing required files: {', '.join(missing)}"
            )
        )

    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise GraphComBODependencyError(
            _setup_message(f"Could not verify the GraphComBO checkout at {root}: {error}")
        ) from error
    if revision != GRAPHCOMBO_COMMIT:
        raise GraphComBODependencyError(
            _setup_message(
                f"GraphComBO is at {revision}, but this integration requires "
                f"{GRAPHCOMBO_COMMIT}."
            )
        )

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    for name in list(sys.modules):
        if name == "search" or name.startswith("search."):
            del sys.modules[name]
    try:
        torch = importlib.import_module("torch")
        transforms = importlib.import_module("botorch.utils.transforms")
        models = importlib.import_module("search.models")
        contexts = importlib.import_module("search.self_combograph")
        trust_region = importlib.import_module("search.trust_region")
    except (ImportError, AttributeError) as error:
        raise GraphComBODependencyError(
            _setup_message(
                "The pinned GraphComBO checkout could not be imported with the installed "
                f"Python dependencies: {error}"
            )
        ) from error

    return _GraphComBOAPI(
        torch=torch,
        standardize=transforms.standardize,
        initialize_model=models.initialize_model,
        get_acqf=models.get_acqf,
        construct_context=contexts.ComboSubgraph_Constructor,
        TrustRegionState=trust_region.TrustRegionState,
        update_state=trust_region.update_state,
    )


def binary_base_graph(matrix_file):
    data = np.load(matrix_file, allow_pickle=False)
    names = data["candidate_names"].astype(str)
    index = {name: i for i, name in enumerate(names)}
    graph = nx.Graph()
    graph.add_nodes_from(range(len(names)))
    graph.add_edges_from(
        (index[u], index[v]) for u, v in data["candidate_graph_edges"].astype(str)
    )
    return graph


def placement_coverage_kernel_graph(
    matrix_file,
    alpha: float = 4.0,
    bandwidth: float = 1.0,
    normalize_distance: bool = True,
):
    """Create a physical graph carrying nearest-sensor coverage features."""
    if alpha < 0 or bandwidth <= 0:
        raise ValueError("alpha must be nonnegative and bandwidth positive")
    data = np.load(matrix_file, allow_pickle=False)
    names = data["candidate_names"].astype(str)
    index = {name: i for i, name in enumerate(names)}
    edges = data["candidate_graph_edges"].astype(str)
    times = data["travel_times_seconds"].astype(float)
    if len(times) != len(edges) or np.any(times <= 0):
        raise ValueError("travel times must be positive and match candidate edges")

    travel_graph = nx.Graph()
    travel_graph.add_nodes_from(range(len(names)))
    travel_graph.add_weighted_edges_from(
        (index[u], index[v], float(t)) for (u, v), t in zip(edges, times)
    )
    travel_distance = np.asarray(
        nx.floyd_warshall_numpy(travel_graph, weight="weight"), dtype=float
    )
    travel_distance /= float(np.median(times))

    graph = nx.Graph()
    graph.add_nodes_from(range(len(names)))
    graph.add_edges_from((index[u], index[v]) for u, v in edges)
    graph.graph["placement_coverage_travel_time"] = travel_distance
    graph.graph["placement_kernel_alpha"] = float(alpha)
    graph.graph["placement_kernel_bandwidth"] = float(bandwidth)
    graph.graph["placement_kernel_normalize_distance"] = bool(normalize_distance)
    return graph


def _placement_coverage_feature(base_graph, placement):
    indices = np.asarray(sorted(placement), dtype=int)
    values = base_graph.graph["placement_coverage_travel_time"][indices]
    feature = np.min(values, axis=0)
    if not np.all(np.isfinite(feature)):
        raise ValueError("travel-time coverage requires a connected candidate graph")
    return feature


def _placement_neighbors(base_graph, placement):
    selected = set(placement)
    neighbors = set()
    for removed in placement:
        for replacement in base_graph.neighbors(removed):
            if replacement not in selected:
                neighbors.add(tuple(sorted(selected - {removed} | {replacement})))
    return sorted(neighbors)


def calibrate_placement_coverage_distance_scale(
    base_graph, sensor_count: int, samples: int = 20_000, seed: int = 0
):
    """Estimate a fixed, objective-independent legal-swap distance scale."""
    if "placement_coverage_travel_time" not in base_graph.graph:
        raise ValueError("placement coverage features are missing")
    if not 0 < sensor_count < len(base_graph):
        raise ValueError("sensor count must be between zero and the graph size")
    rng = np.random.default_rng(seed)
    distances = []
    nodes = np.asarray(sorted(base_graph.nodes), dtype=int)
    while len(distances) < samples:
        placement = tuple(sorted(rng.choice(nodes, sensor_count, replace=False)))
        candidates = _placement_neighbors(base_graph, placement)
        if not candidates:
            continue
        swapped = candidates[int(rng.integers(len(candidates)))]
        left = _placement_coverage_feature(base_graph, placement)
        right = _placement_coverage_feature(base_graph, swapped)
        distances.append(float(np.linalg.norm(left - right)))
    scale = float(np.median(distances))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError("legal-swap coverage distance scale is not positive")
    base_graph.graph["placement_coverage_distance_scale"] = scale
    base_graph.graph["placement_coverage_distance_scale_samples"] = int(samples)
    return scale


def _apply_placement_coverage_weights(combo_graph, base_graph):
    if "placement_coverage_travel_time" not in base_graph.graph:
        return combo_graph
    alpha = base_graph.graph["placement_kernel_alpha"]
    bandwidth = base_graph.graph["placement_kernel_bandwidth"]
    features = {
        placement: _placement_coverage_feature(base_graph, placement)
        for placement in combo_graph.nodes
    }
    edge_distances = {
        (left, right): float(np.linalg.norm(features[left] - features[right]))
        for left, right in combo_graph.edges
    }
    normalize = base_graph.graph.get("placement_kernel_normalize_distance", False)
    scale = base_graph.graph.get("placement_coverage_distance_scale")
    if normalize and scale is None and edge_distances:
        positive = [value for value in edge_distances.values() if value > 0]
        scale = float(np.median(positive)) if positive else 1.0
    distance_scale = float(scale if normalize and scale is not None else 1.0)
    combo_graph.graph["placement_coverage_distance_scale"] = distance_scale
    for (left, right), distance in edge_distances.items():
        z = distance / distance_scale
        similarity = float(np.exp(-(z * z) / (2.0 * bandwidth * bandwidth)))
        combo_graph[left][right]["weight"] = 1.0 + alpha * similarity
    return combo_graph


def _context(api, base_graph, center, settings, seed):
    random.seed(seed)
    combo_graph, radius = api.construct_context(
        base_graph,
        np.asarray(center, dtype=int).reshape(1, -1),
        nx.Graph(),
        Q=settings["context_size"],
        l_max=settings["max_radius"],
        fast_computation=False,
    )
    return _apply_placement_coverage_weights(combo_graph, base_graph), radius


def _training_data(api, observed, context):
    placements = [placement for placement in observed if placement in context]
    x = api.torch.tensor(placements, dtype=api.torch.float32)
    y = -api.torch.tensor(
        [observed[placement] for placement in placements], dtype=api.torch.float32
    ).reshape(-1, 1)
    return placements, x, y


def _select_candidate(api, context, x_train, y_train, settings, cached_eigenbasis):
    nodes = list(context.nodes)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    index_to_node = {index: node for node, index in node_to_index.items()}
    mapped = api.torch.tensor(
        [node_to_index[tuple(row.int().tolist())] for row in x_train],
        dtype=api.torch.float32,
    ).reshape(-1, 1)
    standardized = api.standardize(y_train) if len(y_train) > 1 else api.torch.zeros_like(y_train)
    model, _, eigenbasis = api.initialize_model(
        train_X=mapped,
        train_Y=standardized,
        context_graph=context,
        covar_type="diffusion",
        covar_kwargs={"order": len(context)},
        ard=False,
        fit_model=True,
        cached_eigenbasis=cached_eigenbasis,
        optim_kwargs={
            "train_iters": settings["model_train_iterations"],
            "lr": settings["model_learning_rate"],
        },
        standardize_y=False,
    )
    acquisition = api.get_acqf(
        model, mapped, standardized, acq_type=settings["acquisition"]
    )
    visited = set(int(value) for value in mapped.flatten().tolist())
    available = [index for index in range(len(nodes)) if index not in visited]
    if not available:
        return None, eigenbasis, ""
    candidates = api.torch.tensor(available, dtype=mapped.dtype).reshape(-1, 1)
    values = acquisition(candidates.unsqueeze(1)).reshape(-1)
    chosen = available[int(api.torch.argmax(values))]
    beta = float(model.covar_module.base_kernel.lengthscale.mean().detach())
    return tuple(index_to_node[chosen]), eigenbasis, beta


def run_original_graphcombo(
    oracle: DetectionDelayOracle,
    base_graph: nx.Graph,
    budget: int,
    seed: int,
    initial,
    settings: dict,
    method_name: str = "graphcombo_original_binary",
    graphcombo_root: str | Path | None = None,
):
    """Run the water-sensor experiment using the external GraphComBO modules."""
    api = load_graphcombo(graphcombo_root)
    api.torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 50_000)
    initial = [tuple(sorted(map(int, placement))) for placement in initial]
    if budget < len(initial):
        raise ValueError("budget must be at least the number of initial placements")

    observed = {}
    rows = []
    best_placement = initial[0]
    best_value = float("inf")
    for placement in initial:
        value = oracle.evaluate(np.asarray(placement))
        observed[placement] = value
        if value < best_value:
            best_placement, best_value = placement, value
        rows.append(
            _record(
                oracle, method_name, seed, len(rows) + 1, placement, value,
                best_placement, best_value,
            )
        )

    center = best_placement
    combo_graph, radius = _context(api, base_graph, center, settings, seed)
    state = api.TrustRegionState(
        dim=1,
        n_nodes=settings["context_size"],
        n_nodes_min=5,
        n_nodes_max=settings["context_size"],
        fail_tol=settings["fail_tolerance"],
        succ_tol=settings["success_tolerance"],
        shrink_tol=settings["shrink_tolerance"],
        best_value=-best_value,
    )
    cached_eigenbasis = None
    restart_pending = False
    stagnation = 0
    restart_patience = int(settings["global_restart_patience"])
    relative_tolerance = float(settings["significant_improvement_relative"])

    while len(rows) < budget:
        _, x_train, y_train = _training_data(api, observed, combo_graph)
        if restart_pending:
            placement = tuple(
                sorted(rng.choice(len(oracle), len(initial[0]), replace=False))
            )
            while placement in observed:
                placement = tuple(
                    sorted(rng.choice(len(oracle), len(initial[0]), replace=False))
                )
            beta = "restart"
        else:
            placement, cached_eigenbasis, beta = _select_candidate(
                api, combo_graph, x_train, y_train, settings, cached_eigenbasis
            )
            if placement is None:
                restart_pending = True
                continue

        value = oracle.evaluate(np.asarray(placement))
        observed[placement] = value
        new_y = api.torch.tensor([[-value]], dtype=api.torch.float32)
        old_best = best_value
        improved = value < best_value
        significant = value < old_best - relative_tolerance * max(abs(old_best), 1.0)
        if improved:
            best_placement, best_value = placement, value
            center = placement
        rows.append(
            _record(
                oracle, method_name, seed, len(rows) + 1, placement, value,
                best_placement, best_value, len(combo_graph), len(x_train), beta,
            )
        )

        if restart_pending:
            center = placement
            combo_graph, radius = _context(
                api, base_graph, center, settings, seed + len(rows)
            )
            state = api.TrustRegionState(
                dim=1,
                n_nodes=settings["context_size"],
                n_nodes_min=5,
                n_nodes_max=settings["context_size"],
                fail_tol=settings["fail_tolerance"],
                succ_tol=settings["success_tolerance"],
                shrink_tol=settings["shrink_tolerance"],
                best_value=-value,
            )
            cached_eigenbasis = None
            restart_pending = False
            stagnation = 0
            continue

        state = api.update_state(state, new_y)
        stagnation = 0 if significant else stagnation + 1
        if stagnation >= restart_patience or state.restart_triggered:
            restart_pending = True
            state.restart_triggered = False
        elif improved:
            combo_graph, radius = _context(
                api, base_graph, center, settings, seed + len(rows)
            )
            cached_eigenbasis = None
            state.failure_counter = 0
        elif state.shrink_triggered:
            radius = max(1, radius - 1)
            combo_graph = nx.ego_graph(combo_graph, center, radius)
            cached_eigenbasis = None
            state.shrink_triggered = False

    return rows


run_graph_bayesian_optimization = run_original_graphcombo
