"""五子連線判定（自由五子棋：五子以上連線即勝，不含連珠禁手）。"""

from __future__ import annotations

from .board import BLACK, EMPTY, SIZE, WHITE, WIN_LENGTH, Board, in_bounds, to_index

# 水平、垂直、正斜線(\)、反斜線(/)
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def line_length(board: Board, row: int, col: int, color: int, dr: int, dc: int) -> int:
    """以 (row, col) 為中心，沿 (dr, dc) 正反兩方向計算同色連續棋子數（含自身）。"""
    count = 1
    for sign in (1, -1):
        r, c = row + sign * dr, col + sign * dc
        while in_bounds(r, c) and board.cells[to_index(r, c)] == color:
            count += 1
            r += sign * dr
            c += sign * dc
    return count


def is_winning_move(board: Board, row: int, col: int, color: int) -> bool:
    """假設 (row, col) 已放上 color 的棋子，判斷是否形成五連。"""
    return any(line_length(board, row, col, color, dr, dc) >= WIN_LENGTH for dr, dc in DIRECTIONS)


def would_win(board: Board, idx: int, color: int) -> bool:
    """若 color 下在空位 idx 是否立即獲勝（不修改棋盤）。"""
    if board.cells[idx] != EMPTY:
        return False
    board.cells[idx] = color
    try:
        r, c = divmod(idx, SIZE)
        return is_winning_move(board, r, c, color)
    finally:
        board.cells[idx] = EMPTY


def find_winner(board: Board) -> int | None:
    """完整掃描整個棋盤，回傳獲勝顏色；無人五連則回傳 None。

    用於重播驗證；若雙方同時有五連（不合法盤面）會丟出例外。
    """
    winners = set()
    for r in range(SIZE):
        for c in range(SIZE):
            color = board.cells[to_index(r, c)]
            if color == EMPTY:
                continue
            for dr, dc in DIRECTIONS:
                # 只從線段起點開始數，避免重複
                pr, pc = r - dr, c - dc
                if in_bounds(pr, pc) and board.cells[to_index(pr, pc)] == color:
                    continue
                n, rr, cc = 0, r, c
                while in_bounds(rr, cc) and board.cells[to_index(rr, cc)] == color:
                    n += 1
                    rr += dr
                    cc += dc
                if n >= WIN_LENGTH:
                    winners.add(color)
    if len(winners) > 1:
        raise ValueError("不合法盤面：雙方皆有五連")
    return winners.pop() if winners else None


def _build_windows() -> list[tuple[int, ...]]:
    """所有長度為 5 的直線視窗（共 60 個），供啟發式評估使用。"""
    windows = []
    for r in range(SIZE):
        for c in range(SIZE):
            for dr, dc in DIRECTIONS:
                er, ec = r + dr * (WIN_LENGTH - 1), c + dc * (WIN_LENGTH - 1)
                if in_bounds(er, ec):
                    windows.append(tuple(to_index(r + dr * k, c + dc * k) for k in range(WIN_LENGTH)))
    return windows


WINDOWS: list[tuple[int, ...]] = _build_windows()
CELL_WINDOWS: list[list[tuple[int, ...]]] = [[w for w in WINDOWS if i in w] for i in range(SIZE * SIZE)]

__all__ = [
    "BLACK", "WHITE", "DIRECTIONS", "WINDOWS", "CELL_WINDOWS",
    "line_length", "is_winning_move", "would_win", "find_winner",
]
