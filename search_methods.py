"""Shared initialization and local search for sensor placements."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from random_search import DetectionDelayOracle

Placement = tuple[int, ...]


@dataclass
class CandidateGraph:
    neighbors: list[list[tuple[int, float]]]

    @classmethod
    def from_matrix(cls, matrix_file):
        data = np.load(matrix_file, allow_pickle=False)
        names = data["candidate_names"].astype(str)
        name_to_index = {name: i for i, name in enumerate(names)}
        edges = data["candidate_graph_edges"].astype(str)
        neighbors = [[] for _ in names]
        for u_name, v_name in edges:
            u, v = name_to_index[u_name], name_to_index[v_name]
            neighbors[u].append((v, 1.0))
            neighbors[v].append((u, 1.0))
        return cls(neighbors)

    def placement_neighbors(self, placement: Placement):
        selected = set(placement)
        result = {}
        for old in placement:
            for new, weight in self.neighbors[old]:
                if new in selected:
                    continue
                candidate = tuple(sorted((selected - {old}) | {new}))
                result[candidate] = max(result.get(candidate, 0.0), weight)
        return result


def initial_placements(n_candidates: int, k: int, count: int, seed: int):
    rng = np.random.default_rng(seed)
    placements = []
    seen = set()
    while len(placements) < count:
        placement = tuple(sorted(rng.choice(n_candidates, k, replace=False)))
        if placement not in seen:
            seen.add(placement)
            placements.append(placement)
    return placements


def _record(
    oracle, method, seed, query, placement, value, best_placement, best_value,
    context_size="", training_count="", kernel_beta="",
):
    return {
        "method": method,
        "seed": seed,
        "query": query,
        "objective_hours": value,
        "best_so_far_hours": best_value,
        "placement": "|".join(oracle.candidate_names[list(placement)]),
        "best_placement": "|".join(oracle.candidate_names[list(best_placement)]),
        "context_size": context_size,
        "training_count": training_count,
        "kernel_beta": kernel_beta,
    }


def run_local_search(oracle, graph, k, budget, seed, initial, settings=None):
    settings = settings or {}
    restart_patience = int(settings.get("global_restart_patience", 20))
    relative_tolerance = float(settings.get("significant_improvement_relative", 1e-3))
    rng = np.random.default_rng(seed + 10_000)
    queried = {}
    rows = []
    best_placement = initial[0]
    best_value = float("inf")
    for placement in initial:
        value = oracle.evaluate(np.asarray(placement))
        queried[placement] = value
        if value < best_value:
            best_placement, best_value = placement, value
        rows.append(_record(oracle, "local_search", seed, len(rows)+1, placement,
                            value, best_placement, best_value))

    center_placement = best_placement
    center_value = best_value
    stagnation = 0

    while len(rows) < budget:
        candidates = [
            p for p in graph.placement_neighbors(center_placement) if p not in queried
        ]
        restarting = stagnation >= restart_patience or not candidates
        if not restarting:
            placement = candidates[int(rng.integers(len(candidates)))]
        else:
            placement = initial_placements(len(oracle), k, 1, int(rng.integers(2**31)))[0]
            while placement in queried:
                placement = initial_placements(len(oracle), k, 1, int(rng.integers(2**31)))[0]
        value = oracle.evaluate(np.asarray(placement))
        queried[placement] = value
        old_best = best_value
        if value < best_value:
            best_placement, best_value = placement, value
        significant = value < old_best - relative_tolerance * max(abs(old_best), 1.0)
        stagnation = 0 if significant or restarting else stagnation + 1
        if restarting or value < center_value:
            center_placement, center_value = placement, value
        rows.append(_record(oracle, "local_search", seed, len(rows)+1, placement,
                            value, best_placement, best_value))
    return rows
