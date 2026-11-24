import numpy as np
import pytest

from original_graphcombo import (
    GraphComBODependencyError,
    binary_base_graph,
    calibrate_placement_coverage_distance_scale,
    load_graphcombo,
    placement_coverage_kernel_graph,
    _apply_placement_coverage_weights,
)


def test_binary_base_graph_uses_saved_candidate_edges(tmp_path):
    matrix = tmp_path / "matrix.npz"
    np.savez(
        matrix,
        candidate_names=np.array(["A", "B", "C"]),
        candidate_graph_edges=np.array([["A", "B"], ["B", "C"]]),
    )
    graph = binary_base_graph(matrix)
    assert set(graph.nodes) == {0, 1, 2}
    assert set(graph.edges) == {(0, 1), (1, 2)}


def test_placement_coverage_kernel_reweights_combo_edges(tmp_path):
    matrix = tmp_path / "matrix.npz"
    np.savez(
        matrix,
        candidate_names=np.array(["A", "B", "C"]),
        candidate_graph_edges=np.array([["A", "B"], ["B", "C"]]),
        travel_times_seconds=np.array([1.0, 5.0]),
    )
    base = placement_coverage_kernel_graph(
        matrix, alpha=2.0, bandwidth=1.0, normalize_distance=True
    )
    combo = __import__("networkx").Graph()
    combo.add_edges_from([((0,), (1,)), ((1,), (2,))])
    _apply_placement_coverage_weights(combo, base)
    distances = []
    for left, right in combo.edges:
        travel_time = base.graph["placement_coverage_travel_time"]
        left_feature = np.min(travel_time[np.asarray(left)], axis=0)
        right_feature = np.min(travel_time[np.asarray(right)], axis=0)
        distances.append(np.linalg.norm(left_feature - right_feature))
    assert np.isclose(
        combo.graph["placement_coverage_distance_scale"], np.median(distances)
    )
    assert combo[(0,)][(1,)]["weight"] != combo[(1,)][(2,)]["weight"]
    assert all(data["weight"] > 1.0 for _, _, data in combo.edges(data=True))


def test_placement_coverage_scale_is_fixed_across_contexts(tmp_path):
    matrix = tmp_path / "matrix.npz"
    np.savez(
        matrix,
        candidate_names=np.array(["A", "B", "C", "D"]),
        candidate_graph_edges=np.array([["A", "B"], ["B", "C"], ["C", "D"]]),
        travel_times_seconds=np.array([1.0, 2.0, 3.0]),
    )
    base = placement_coverage_kernel_graph(
        matrix, alpha=2.0, bandwidth=1.0, normalize_distance=True
    )
    scale = calibrate_placement_coverage_distance_scale(
        base, sensor_count=2, samples=100, seed=3
    )
    first = __import__("networkx").Graph([((0, 1), (1, 2))])
    second = __import__("networkx").Graph([((1, 2), (2, 3))])
    _apply_placement_coverage_weights(first, base)
    _apply_placement_coverage_weights(second, base)
    assert first.graph["placement_coverage_distance_scale"] == scale
    assert second.graph["placement_coverage_distance_scale"] == scale


def test_travel_time_coverage_uses_nearest_sensor_distance(tmp_path):
    matrix = tmp_path / "matrix.npz"
    np.savez(
        matrix,
        candidate_names=np.array(["A", "B", "C"]),
        candidate_graph_edges=np.array([["A", "B"], ["B", "C"]]),
        travel_times_seconds=np.array([1.0, 3.0]),
    )
    base = placement_coverage_kernel_graph(matrix)
    feature = __import__("original_graphcombo")._placement_coverage_feature(
        base, (0, 2)
    )
    assert np.allclose(feature, [0.0, 0.5, 0.0])


def test_external_graphcombo_has_actionable_missing_dependency_error(monkeypatch):
    monkeypatch.delenv("GRAPHCOMBO_ROOT", raising=False)
    with pytest.raises(GraphComBODependencyError, match="git clone"):
        load_graphcombo()


def test_external_graphcombo_rejects_incomplete_checkout(tmp_path):
    with pytest.raises(GraphComBODependencyError, match="missing required files"):
        load_graphcombo(tmp_path)
