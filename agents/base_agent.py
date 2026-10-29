"""所有對弈代理（AI / 對手）的統一介面。

對弈流程（training/trainer.py 的 play_game）呼叫順序：
    start_game(color, seed)
    重複：select_move(board, color) / observe_move(board_before, move, mover_color)
    end_game(winner, my_color)

學習型 AI 與對手策略完全分離：對手只透過 select_move 落子，
不會讀取或修改學習 AI 的任何資料。
"""

from __future__ import annotations

import random

from core.board import Board


class Agent:
    kind = "base"
    version = "1.0"

    def __init__(self, name: str | None = None, seed: int | None = None):
        self.name = name or self.kind
        self.rng = random.Random(seed)

    # ---- 生命週期 ---------------------------------------------------------
    def start_game(self, color: int, seed: int | None = None) -> None:
        """新局開始。若給定 seed，重設亂數以利重現。"""
        self.color = color
        if seed is not None:
            self.rng.seed(seed)

    def select_move(self, board: Board, color: int) -> int:
        """回傳一個合法空位的一維索引。"""
        raise NotImplementedError

    def observe_move(self, board_before: Board, move: int, color: int) -> None:
        """每一手（雙方）落子後都會通知；預設不做事。"""

    def end_game(self, winner: int | None, my_color: int) -> dict | None:
        """棋局結束；學習型代理可在此更新權重並回傳學習紀錄。"""
        return None

    # ---- 描述 -------------------------------------------------------------
    def params(self) -> dict:
        return {}

    def describe(self) -> dict:
        return {"type": self.kind, "name": self.name, "version": self.version, "params": self.params()}


def weighted_choice(rng: random.Random, weights: list[float]) -> int:
    """依權重抽取索引：P(i) = w_i / Σw。權重總和必須 > 0。"""
    total = sum(weights)
    if total <= 0:
        raise ValueError("權重總和必須 > 0")
    x = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if x < acc:
            return i
    return max(i for i, w in enumerate(weights) if w > 0)  # 浮點誤差保護：回傳最後一個正權重
