"""第三版 AI（第二次修正）：棋形計數牌＋按比例增減牌數。

與第二版（pattern_menace_agent.py）相同的地方
    - 每個空位的牌數 = 四個方向棋形牌數的乘積，依牌數比例抽籤；
    - 每局結束後依勝負調整本局用過的牌。

兩個修正（見 docs/檢討報告.md 第 10 節）
1. 棋形改用「視窗計數」（features="window"，見 pattern_menace_agent.window_keys）：
   每個方向只記「自己在未被擋的五格內最多幾子、對手最多幾子」，共約 30 種牌，
   每一局都會大量重複出現，經驗才累積得起來。
2. 牌數改為「按比例」增減，並考慮同一局面的其他候選位置（策略梯度 REINFORCE）：
       牌數 = initial_weight × e^θ，P(a|s) ∝ Π_d 牌[棋形_d(s,a)]
       本局第 t 手（共 n 手）的調整量 g_t = α × R × γ^(n-1-t)，R = 勝 +1 / 敗 −1 / 和 0
       被選中位置用到的棋形牌：θ += g_t
       該局面每個候選位置用到的棋形牌：θ −= g_t × P(候選)
   效果：輸棋時「選了的」按比例減少，「沒選的」相對增加，所以 AI 能學到「該擋卻沒擋」；
   越接近終局的手調整越大（γ < 1）。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

from core.board import BLACK, Board, to_rc
from core.symmetry import encode, relative_cells
from training.memory import OpponentPatternMemory, ThetaCardMemory
from training.rewards import AI_LOSS, AI_WIN, DRAW

from .base_agent import Agent, weighted_choice
from .pattern_menace_agent import pattern_keys, window_keys

FORMAT = "card-pg-weights"
FORMAT_VERSION = 1


@dataclass
class GradientCardConfig:
    initial_weight: float = 10.0   # 未學過的棋形牌數（只影響顯示，不影響機率）
    alpha: float = 0.1             # 學習率：每次調整的幅度（θ 加減量）
    gamma: float = 0.9             # 越早的手調整越小：第 t 手乘上 γ^(n-1-t)
    win_return: float = 1.0
    loss_return: float = -1.0
    draw_return: float = 0.0
    features: str = "window"       # "window" 或 "line"（第二版的完整排列，用於對照實驗）
    version: str = "card-pg-1.0"

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "GradientCardConfig":
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if cfg.features not in ("window", "line"):
            raise ValueError("features 必須是 'window' 或 'line'")
        return cfg


class GradientCardAgent(Agent):
    kind = "card_pg"

    def __init__(self, config: GradientCardConfig | None = None, name: str | None = None,
                 seed: int | None = None, learning: bool = True):
        super().__init__(name or "card-pg", seed)
        self.config = config or GradientCardConfig()
        self.memory = ThetaCardMemory(self.config.initial_weight)
        self.opponent_memory = OpponentPatternMemory()
        self.learning = learning
        self.games_trained = 0
        self.steps: list[dict] = []
        self._expect: list[dict[str, float]] = []
        self.color = BLACK

    @property
    def version(self) -> str:  # type: ignore[override]
        return self.config.version

    def params(self) -> dict:
        return {**self.config.to_dict(), "learning": self.learning, "games_trained": self.games_trained}

    def start_game(self, color: int, seed: int | None = None) -> None:
        super().start_game(color, seed)
        self.steps = []
        self._expect = []

    def _keys(self, cells: list[int], a: int, color: int) -> list[str]:
        if self.config.features == "window":
            return window_keys(cells, a, color)
        return pattern_keys(cells, a, color)

    def deck(self, board: Board, color: int) -> tuple[list[int], list[list[str]], list[float]]:
        """回傳 (合法位置, 各位置四個棋形, 各位置被抽中的機率)。"""
        legal = board.legal_moves()
        keys = [self._keys(board.cells, a, color) for a in legal]
        logits = [sum(self.memory.get_theta(k) for k in ks) for ks in keys]
        m = max(logits)
        ex = [math.exp(x - m) for x in logits]
        z = sum(ex)
        return legal, keys, [e / z for e in ex]

    def move_probabilities(self, board: Board, color: int) -> dict[int, float]:
        legal, _, probs = self.deck(board, color)
        return dict(zip(legal, probs))

    def select_move(self, board: Board, color: int) -> int:
        legal, keys, probs = self.deck(board, color)
        if not legal:
            raise ValueError("沒有合法落子位置")
        i = weighted_choice(self.rng, probs)
        a = legal[i]
        if self.learning:
            expect: dict[str, float] = {}
            for ks, p in zip(keys, probs):
                for k in ks:
                    expect[k] = expect.get(k, 0.0) + p
            self._expect.append(expect)
        self.steps.append({
            "move_no": board.stone_count() + 1,
            "state": encode(relative_cells(board.cells, color)),
            "action": list(to_rc(a)),
            "patterns": keys[i],
            "cards_before": [round(self.memory.card(k), 3) for k in keys[i]],
            "prob": round(probs[i], 6),
            "deck_size": len(legal),
        })
        return a

    def observe_move(self, board_before: Board, move: int, color: int) -> None:
        if color == self.color or not self.learning:
            return
        legal = board_before.legal_moves()
        available = [self._keys(board_before.cells, a, color) for a in legal]
        self.opponent_memory.record(available, legal.index(move))

    def end_game(self, winner: int | None, my_color: int) -> dict:
        result = DRAW if winner is None else (AI_WIN if winner == my_color else AI_LOSS)
        if self.learning:
            c = self.config
            ret = {AI_WIN: c.win_return, AI_LOSS: c.loss_return, DRAW: c.draw_return}[result]
            n = len(self.steps)
            for t, (step, expect) in enumerate(zip(self.steps, self._expect)):
                g = c.alpha * ret * c.gamma ** (n - 1 - t)
                for k, p in expect.items():
                    self.memory.add(k, -g * p)
                for k in step["patterns"]:
                    self.memory.add(k, g)
                    self.memory.uses[k] = self.memory.uses.get(k, 0) + 1
                step["delta_theta"] = round(g, 6)
            for step in self.steps:
                step["cards_after"] = [round(self.memory.card(k), 3) for k in step["patterns"]]
            self.games_trained += 1
        return {"result": result, "steps": self.steps}

    # ---- 自我對弈 / 保存 ---------------------------------------------------
    def twin(self, name: str | None = None) -> "GradientCardAgent":
        """自我對弈用：共用同一副牌與對手記憶的另一個座位。"""
        other = GradientCardAgent(self.config, name=name or f"{self.name}-self", learning=self.learning)
        other.memory = self.memory
        other.opponent_memory = self.opponent_memory
        other.kind = "self_play"
        return other

    def to_dict(self) -> dict:
        return {"format": FORMAT, "format_version": FORMAT_VERSION, "config": self.config.to_dict(),
                "games_trained": self.games_trained, "memory": self.memory.to_dict(),
                "opponent_memory": self.opponent_memory.to_dict()}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)

    @classmethod
    def from_dict(cls, data: dict, learning: bool = True, seed: int | None = None,
                  name: str | None = None) -> "GradientCardAgent":
        if data.get("format") != FORMAT:
            raise ValueError("不是第三版位子牌權重檔")
        agent = cls(GradientCardConfig.from_dict(data["config"]), name=name, seed=seed, learning=learning)
        agent.memory = ThetaCardMemory.from_dict(data["memory"])
        agent.opponent_memory = OpponentPatternMemory.from_dict(data.get("opponent_memory", {}))
        agent.games_trained = data.get("games_trained", 0)
        return agent

    @classmethod
    def load(cls, path: str, learning: bool = True, seed: int | None = None,
             name: str | None = None) -> "GradientCardAgent":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f), learning=learning, seed=seed, name=name)

    def clone(self, learning: bool | None = None) -> "GradientCardAgent":
        return GradientCardAgent.from_dict(json.loads(json.dumps(self.to_dict())),
                                           learning=self.learning if learning is None else learning,
                                           name=self.name)
