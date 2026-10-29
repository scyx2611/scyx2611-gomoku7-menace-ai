"""7x7 棋盤資料結構與落子合法性。

棋盤以「絕對顏色」儲存：
    EMPTY = 0, BLACK = 1（先手）, WHITE = -1（後手）

AI 在建立「局面識別碼」時，會再轉成「相對於自己」的表示
（自己 = 1、對手 = -1，見 core/symmetry.py 與 agents/menace_agent.py），
因此同一套學習資料可以比較先手與後手。

位置可用 (row, col) 或一維索引 idx = row * SIZE + col 表示。
"""

from __future__ import annotations

SIZE = 7
WIN_LENGTH = 5
NUM_CELLS = SIZE * SIZE

EMPTY = 0
BLACK = 1
WHITE = -1

COLOR_NAME = {BLACK: "black", WHITE: "white"}
NAME_COLOR = {"black": BLACK, "white": WHITE}


class IllegalMoveError(ValueError):
    """嘗試在非法位置落子（越界、已有棋子、棋局已結束）。"""


def to_index(row: int, col: int) -> int:
    return row * SIZE + col


def to_rc(idx: int) -> tuple[int, int]:
    return divmod(idx, SIZE)


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < SIZE and 0 <= col < SIZE


class Board:
    """7x7 棋盤。cells 為長度 49 的 list。"""

    __slots__ = ("cells",)

    def __init__(self, cells: list[int] | None = None):
        if cells is None:
            self.cells = [EMPTY] * NUM_CELLS
        else:
            if len(cells) != NUM_CELLS:
                raise ValueError(f"cells 長度必須為 {NUM_CELLS}")
            self.cells = list(cells)

    # ---- 查詢 -------------------------------------------------------------
    def get(self, row: int, col: int) -> int:
        if not in_bounds(row, col):
            raise IndexError(f"({row}, {col}) 超出棋盤範圍")
        return self.cells[to_index(row, col)]

    def is_empty(self, row: int, col: int) -> bool:
        return in_bounds(row, col) and self.cells[to_index(row, col)] == EMPTY

    def legal_moves(self) -> list[int]:
        """所有空位的一維索引（由小到大）。"""
        return [i for i, v in enumerate(self.cells) if v == EMPTY]

    def is_full(self) -> bool:
        return EMPTY not in self.cells

    def stone_count(self) -> int:
        return NUM_CELLS - self.cells.count(EMPTY)

    # ---- 修改 -------------------------------------------------------------
    def place(self, row: int, col: int, color: int) -> None:
        if color not in (BLACK, WHITE):
            raise ValueError(f"不合法的棋子顏色: {color}")
        if not in_bounds(row, col):
            raise IllegalMoveError(f"({row}, {col}) 超出棋盤範圍")
        idx = to_index(row, col)
        if self.cells[idx] != EMPTY:
            raise IllegalMoveError(f"({row}, {col}) 已有棋子")
        self.cells[idx] = color

    def reset(self) -> None:
        self.cells = [EMPTY] * NUM_CELLS

    def copy(self) -> "Board":
        return Board(self.cells)

    def rows(self) -> list[list[int]]:
        return [self.cells[r * SIZE:(r + 1) * SIZE] for r in range(SIZE)]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Board) and self.cells == other.cells

    def __hash__(self) -> int:
        return hash(tuple(self.cells))

    def render(self, last_move: int | None = None) -> str:
        """文字棋盤：X = 黑（先手）、O = 白（後手）、. = 空。"""
        symbols = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        lines = ["   " + " ".join(str(c) for c in range(SIZE))]
        for r in range(SIZE):
            row = []
            for c in range(SIZE):
                s = symbols[self.cells[to_index(r, c)]]
                if last_move is not None and to_index(r, c) == last_move:
                    s = s.lower() if s != "." else s
                row.append(s)
            lines.append(f"{r}  " + " ".join(row))
        return "\n".join(lines)

    __str__ = render
