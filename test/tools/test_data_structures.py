import pytest

from tools.data_structures import DSU


def test_dsu_init():
    n = 10
    dsu = DSU(n)
    dsu.new_set()
    assert dsu.n == n + 1
    for i in range(n + 1):
        assert dsu.find_root(i) == i


def test_dsu_union():
    n = 10
    dsu = DSU(n)
    dsu.union_sets(0, 9)
    assert dsu.find_root(0) == dsu.find_root(9)

    dsu.union_sets(1, 3)
    dsu.union_sets(0, 3)
    root = dsu.find_root(0)
    assert all(dsu.find_root(i) == root for i in [0, 1, 3, 9])

    dsu.union_sets(2, 7)
    dsu.union_sets(5, 6)
    dsu.union_sets(2, 8)

    # at this point the sets:
    sets = [
        [0, 1, 3, 9],
        [2, 7, 8],
        [5, 6],
        [4],
    ]
    for s in sets:
        root = dsu.find_root(s[0])
        assert all(dsu.find_root(i) == root for i in s)

    dsu.union_sets(0, 8)
    dsu.union_sets(5, 4)

    # at this point the sets:
    sets = [
        [0, 1, 2, 3, 7, 8, 9],
        [4, 5, 6],
    ]

    for s in sets:
        root = dsu.find_root(s[0])
        assert all(dsu.find_root(i) == root for i in s)

    dsu.union_sets(0, 5)
    root = dsu.find_root(0)
    assert all(dsu.find_root(i) == root for i in range(10))
