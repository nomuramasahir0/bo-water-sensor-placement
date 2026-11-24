import numpy as np

from random_search import DetectionDelayOracle, run_random_search


def test_oracle_and_random_search(tmp_path):
    matrix = tmp_path / "delays.npz"
    np.savez(
        matrix,
        delays_hours=np.array([[1.0, 3.0, 2.0], [4.0, 1.0, 5.0]]),
        candidate_names=np.array(["A", "B", "C"]),
    )
    oracle = DetectionDelayOracle(matrix)
    assert oracle.evaluate(np.array([0, 1])) == 1.0

    rows = run_random_search(oracle, sensor_count=2, budget=3, seed=7)
    assert len(rows) == 3
    assert len({row["placement"] for row in rows}) == 3
    best = [row["best_so_far_hours"] for row in rows]
    assert best == sorted(best, reverse=True)
