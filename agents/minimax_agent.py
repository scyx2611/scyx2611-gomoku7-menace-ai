"""搜尋型對手：Minimax + Alpha-Beta 剪枝（深度限制 + 候選手數限制）。

注意：這是「搜尋」找好棋，不涉及獎懲學習；只用來當作比較基準/測試對手。
"""

from __future__ import annotations

from core.board import Board
from core.rules import WINDOWS, would_win

from .base_agent import Agent
from .heuristic_agent import pattern_scores

WIN_SCORE = 1_000_000
_LINE_VALUE = (0, 1, 10, 100, 1000, WIN_SCORE)


def evaluate(board: Board, color: int) -> int:
    """靜態評估：只計算未被對方阻斷的視窗。"""
    cells = board.cells
    total = 0
    for w in WINDOWS:
        own = opp = 0
        for j in w:
            v = cells[j]
            if v == color:
                own += 1
            elif v == -color:
                opp += 1
        if opp == 0:
            total += _LINE_VALUE[own]
        elif own == 0:
            total -= _LINE_VALUE[opp]
    return total


class MinimaxAgent(Agent):
    kind = "minimax"
    version = "1.0"

    def __init__(self, depth: int = 2, max_candidates: int = 8, name: str | None = None, seed: int | None = None):
        super().__init__(name or f"minimax-d{depth}", seed)
        self.depth = depth
        self.max_candidates = max_candidates

    def params(self) -> dict:
        return {"depth": self.depth, "max_candidates": self.max_candidates}

    def _candidates(self, board: Board, color: int) -> list[int]:
        scores = pattern_scores(board, color)
        ordered = sorted(scores, key=lambda i: (-scores[i], i))
        return ordered[: self.max_candidates]

    def _search(self, board: Board, color: int, depth: int, alpha: float, beta: float) -> float:
        """negamax：回傳以 color 角度的分數。"""
        legal = board.legal_moves()
        if not legal:
            return 0
        for i in legal:
            if would_win(board, i, color):
                return WIN_SCORE + depth  # 越早贏越好
        if depth == 0:
            return evaluate(board, color)
        best = -float("inf")
        for i in self._candidates(board, color):
            board.cells[i] = color
            val = -self._search(board, -color, depth - 1, -beta, -alpha)
            board.cells[i] = 0
            if val > best:
                best = val
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return best

    def select_move(self, board: Board, color: int) -> int:
        work = board.copy()
        legal = work.legal_moves()
        for i in legal:
            if would_win(work, i, color):
                return i
        best_val, best_moves = -float("inf"), []
        for i in self._candidates(work, color):
            work.cells[i] = color
            # 視窗下界設為 best_val - 1，讓「同分」的著手也得到精確值，可正確收集並列最佳手
            alpha = best_val - 1 if best_val != -float("inf") else -float("inf")
            val = -self._search(work, -color, self.depth - 1, -float("inf"), -alpha)
            work.cells[i] = 0
            if val > best_val:
                best_val, best_moves = val, [i]
            elif val == best_val:
                best_moves.append(i)
        return self.rng.choice(best_moves)
