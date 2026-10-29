"""隨機對手：在所有空位中均勻隨機選一個。"""

from __future__ import annotations

from core.board import Board

from .base_agent import Agent


class RandomAgent(Agent):
    kind = "random"

    def select_move(self, board: Board, color: int) -> int:
        return self.rng.choice(board.legal_moves())
