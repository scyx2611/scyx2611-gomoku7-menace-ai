"""棋盤八種對稱（二面體群 D4）與局面標準化（canonical state）。

八種轉換（以 (r, c) -> (r', c') 表示，N = 7, m = N - 1）：
    0 identity          (r, c)
    1 rot90 (順時針)    (c, m - r)
    2 rot180            (m - r, m - c)
    3 rot270            (m - c, r)
    4 flip_horizontal   (r, m - c)      左右鏡像
    5 flip_vertical     (m - r, c)      上下鏡像
    6 transpose         (c, r)          主對角線鏡像
    7 anti_transpose    (m - c, m - r)  副對角線鏡像

標準化做法：把「相對於 AI 的盤面」(自己=1、對手=2、空=0) 做八種轉換，
取字典序最小的字串當作 canonical key，並記下使用的轉換 t。
  - 只改變幾何位置，不交換黑白身分，因此不會把顏色互換的盤面誤判為相同；
  - 由於使用相對表示且盤面上自己/對手的子數決定了輪到誰，
    「輪到誰下」也自然包含在 key 裡；
  - 動作 a（實際座標）轉成標準座標：PERM[t][a]；
    標準座標轉回實際座標：INV_PERM[t][ca]。
"""

from __future__ import annotations

from typing import Sequence

from .board import NUM_CELLS, SIZE, to_index

_M = SIZE - 1

TRANSFORM_NAMES = (
    "identity", "rot90", "rot180", "rot270",
    "flip_horizontal", "flip_vertical", "transpose", "anti_transpose",
)

_FUNCS = (
    lambda r, c: (r, c),
    lambda r, c: (c, _M - r),
    lambda r, c: (_M - r, _M - c),
    lambda r, c: (_M - c, r),
    lambda r, c: (r, _M - c),
    lambda r, c: (_M - r, c),
    lambda r, c: (c, r),
    lambda r, c: (_M - c, _M - r),
)

# PERM[t][i] = 原位置 i 經轉換 t 後的位置
PERM: tuple[tuple[int, ...], ...] = tuple(
    tuple(to_index(*f(*divmod(i, SIZE))) for i in range(NUM_CELLS)) for f in _FUNCS
)
# INV_PERM[t][j] = 轉換後位置 j 對應的原位置
INV_PERM: tuple[tuple[int, ...], ...] = tuple(
    tuple(p.index(j) for j in range(NUM_CELLS)) for p in PERM
)

_CHAR = {0: "0", 1: "1", -1: "2"}


def relative_cells(cells: Sequence[int], color: int) -> list[int]:
    """把絕對顏色轉成相對於 color 的表示：自己 = 1、對手 = -1、空 = 0。"""
    return [v * color for v in cells]


def encode(rel_cells: Sequence[int]) -> str:
    """相對盤面 -> 49 字元字串（0 空、1 自己、2 對手）。"""
    return "".join(_CHAR[v] for v in rel_cells)


def transform_cells(cells: Sequence[int], t: int) -> list[int]:
    out = [0] * NUM_CELLS
    perm = PERM[t]
    for i, v in enumerate(cells):
        out[perm[i]] = v
    return out


def all_symmetries(rel_cells: Sequence[int]) -> list[str]:
    """八種轉換後的編碼字串（依轉換編號排列）。"""
    s = encode(rel_cells)
    return ["".join(map(s.__getitem__, inv)) for inv in INV_PERM]


def canonical(rel_cells: Sequence[int]) -> tuple[str, int]:
    """回傳 (canonical_key, t)。t 為把實際盤面轉成標準盤面所用的轉換。

    若多個轉換得到相同的最小字串（盤面本身有對稱性），取編號最小者，結果固定。
    """
    keys = all_symmetries(rel_cells)
    best_t = min(range(8), key=lambda t: (keys[t], t))
    return keys[best_t], best_t


def to_canonical_action(action: int, t: int) -> int:
    return PERM[t][action]


def from_canonical_action(canon_action: int, t: int) -> int:
    return INV_PERM[t][canon_action]
