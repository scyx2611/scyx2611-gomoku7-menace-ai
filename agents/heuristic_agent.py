"""規則型（啟發式）對手。

level = "basic"：
    1. 能直接五連就下；2. 對手下一手能五連就擋；3. 否則隨機（確定性模式取最靠近中央的空位）。
level = "pattern"（預設）：
    以 60 個長度 5 的直線視窗計分：
    - 進攻：視窗內沒有對手棋子時，依自己子數給分；
    - 防守：視窗內沒有自己棋子時，依對手子數給分；
    再加上微小的中心偏好，取最高分。

deterministic=True 時同分取「最靠近中央、再取索引最小」者（同局面必定同一手），
否則以亂數在同分中選擇；epsilon > 0 時以該機率改下隨機空位。
"""

from __future__ import annotations

from core.board import SIZE, Board
from core.rules import CELL_WINDOWS, would_win

from .base_agent import Agent

ATTACK = (1, 8, 60, 600, 100_000)   # 下這手前，視窗中已有 0..4 顆自己的子
DEFENSE = (0, 5, 45, 450, 50_000)   # 視窗中已有 0..4 顆對手的子
_CENTER = (SIZE - 1) / 2
CENTER_BONUS = [0.5 - (abs(i // SIZE - _CENTER) + abs(i % SIZE - _CENTER)) / 12 for i in range(SIZE * SIZE)]


def pattern_scores(board: Board, color: int) -> dict[int, float]:
    """每個空位的啟發式分數（越高越好）。"""
    cells = board.cells
    scores: dict[int, float] = {}
    for i, v in enumerate(cells):
        if v != 0:
            continue
        s = CENTER_BONUS[i]
        for w in CELL_WINDOWS[i]:
            own = opp = 0
            for j in w:
                cj = cells[j]
                if cj == color:
                    own += 1
                elif cj == -color:
                    opp += 1
            if opp == 0:
                s += ATTACK[own]
            if own == 0:
                s += DEFENSE[opp]
        scores[i] = s
    return scores


class HeuristicAgent(Agent):
    kind = "heuristic"
    version = "1.0"

    def __init__(self, level: str = "pattern", deterministic: bool = False, epsilon: float = 0.0,
                 name: str | None = None, seed: int | None = None):
        if level not in ("basic", "pattern"):
            raise ValueError("level 必須是 'basic' 或 'pattern'")
        super().__init__(name or f"heuristic-{level}", seed)
        self.level = level
        self.deterministic = deterministic
        self.epsilon = epsilon

    def params(self) -> dict:
        return {"level": self.level, "deterministic": self.deterministic, "epsilon": self.epsilon}

    def _pick(self, candidates: list[int]) -> int:
        if self.deterministic:
            return min(candidates, key=lambda i: (-CENTER_BONUS[i], i))
        return self.rng.choice(candidates)

    def select_move(self, board: Board, color: int) -> int:
        legal = board.legal_moves()
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            return self.rng.choice(legal)
        if self.level == "basic":
            wins = [i for i in legal if would_win(board, i, color)]
            if wins:
                return self._pick(wins)
            blocks = [i for i in legal if would_win(board, i, -color)]
            if blocks:
                return self._pick(blocks)
            return self._pick(legal)
        scores = pattern_scores(board, color)
        best = max(scores.values())
        return self._pick([i for i, s in scores.items() if s >= best - 1e-9])
