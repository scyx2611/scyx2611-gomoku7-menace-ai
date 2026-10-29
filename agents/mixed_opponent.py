"""混合對手：每局開始時依權重從對手池中抽一個（例如自我對弈 50%、隨機版規則型 20%…）。

用途：避免學習 AI 只對單一固定對手背出一條路線。
每局實際抽到的對手會寫進棋譜的 opponent.current 欄位。
"""

from __future__ import annotations

from core.board import Board

from .base_agent import Agent, weighted_choice


class MixedOpponent(Agent):
    kind = "mixed"

    def __init__(self, pool: list[tuple[Agent, float]], name: str | None = None, seed: int | None = None):
        if not pool:
            raise ValueError("對手池不可為空")
        super().__init__(name or "mixed", seed)
        self.pool = pool
        self.current: Agent = pool[0][0]

    def start_game(self, color: int, seed: int | None = None) -> None:
        super().start_game(color, seed)
        self.current = self.pool[weighted_choice(self.rng, [w for _, w in self.pool])][0]
        self.current.start_game(color, None if seed is None else seed ^ 0x5F3759DF)

    def select_move(self, board: Board, color: int) -> int:
        return self.current.select_move(board, color)

    def observe_move(self, board_before: Board, move: int, color: int) -> None:
        self.current.observe_move(board_before, move, color)

    def end_game(self, winner: int | None, my_color: int) -> dict | None:
        return self.current.end_game(winner, my_color)

    def params(self) -> dict:
        return {"pool": [{"name": a.name, "type": a.kind, "weight": w} for a, w in self.pool]}

    def describe(self) -> dict:
        return {**super().describe(), "current": self.current.describe()}

    def close(self) -> None:
        for a, _ in self.pool:
            if hasattr(a, "close"):
                a.close()
