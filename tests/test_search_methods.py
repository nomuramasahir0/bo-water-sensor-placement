from search_methods import CandidateGraph


def test_placement_neighbors_are_fixed_cardinality_swaps():
    graph = CandidateGraph(
        [
            [(1, 0.5)],
            [(0, 0.5), (2, 0.8)],
            [(1, 0.8), (3, 0.2)],
            [(2, 0.2)],
        ]
    )
    neighbors = graph.placement_neighbors((0, 2))
    assert neighbors == {(1, 2): 0.5, (0, 1): 0.8, (0, 3): 0.2}
    assert all(len(p) == 2 and len(set(p)) == 2 for p in neighbors)
