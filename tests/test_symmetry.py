"""必要測試 6、7：等價盤面的 canonical state 相同；對稱後的行動座標可正確還原。"""

import random

from core.board import NUM_CELLS, SIZE
from core.symmetry import (INV_PERM, PERM, all_symmetries, canonical, encode, from_canonical_action,
                           to_canonical_action, transform_cells)


def random_rel(rng, n_stones=10):
    cells = [0] * NUM_CELLS
    for i in rng.sample(range(NUM_CELLS), n_stones):
        cells[i] = rng.choice([1, -1])
    return cells


def test_permutations_are_bijections_and_group_closed():
    perms = set(PERM)
    assert len(perms) == 8
    for p in PERM:
        assert sorted(p) == list(range(NUM_CELLS))
    for p in PERM:  # D4 封閉性：任兩個轉換的合成仍在集合中
        for q in PERM:
            assert tuple(q[p[i]] for i in range(NUM_CELLS)) in perms


def test_center_fixed_and_rot90_example():
    center = (SIZE // 2) * SIZE + SIZE // 2
    assert all(p[center] == center for p in PERM)
    # rot90 順時針：(0,0) -> (0,6)
    assert PERM[1][0] == 6


def test_equivalent_boards_share_canonical_key():
    rng = random.Random(1)
    for _ in range(200):
        rel = random_rel(rng, rng.randint(1, 20))
        key, _ = canonical(rel)
        for t in range(8):
            assert canonical(transform_cells(rel, t))[0] == key


def test_color_swap_is_not_merged():
    rel = [0] * NUM_CELLS
    rel[0], rel[1] = 1, -1
    swapped = [-v for v in rel]
    assert canonical(rel)[0] != canonical(swapped)[0] or rel == swapped
    # 只有一顆子時，自己的子與對手的子也不能被視為同一局面
    a = [0] * NUM_CELLS; a[24] = 1
    b = [0] * NUM_CELLS; b[24] = -1
    assert canonical(a)[0] != canonical(b)[0]


def test_action_round_trip_and_consistency():
    rng = random.Random(2)
    for _ in range(200):
        rel = random_rel(rng, rng.randint(0, 20))
        key, t = canonical(rel)
        for a in range(NUM_CELLS):
            ca = to_canonical_action(a, t)
            assert from_canonical_action(ca, t) == a
            # 標準盤面上 ca 位置的內容 = 實際盤面 a 位置的內容
            assert key[ca] == encode(rel)[a]


def test_all_symmetries_matches_transform_cells():
    rng = random.Random(3)
    rel = random_rel(rng, 15)
    syms = all_symmetries(rel)
    for t in range(8):
        assert syms[t] == encode(transform_cells(rel, t))
        assert all(INV_PERM[t][PERM[t][i]] == i for i in range(NUM_CELLS))
