"""修正後 AI：「棋形位子牌」學習 AI。

修正前（menace_agent.py）的位子牌綁定「整個盤面」，7×7 盤面組合太多，
AI 幾乎不會兩次遇到同一個盤面，獎懲無法累積（見檢討報告）。

修正方式：位子牌改為綁定「該位置四個方向的局部棋形」。
    對每個空位 a，取通過 a 的四條線（橫、直、兩斜），
    每條線取 a 左右各 radius 格（預設 4 格，正好涵蓋所有可能的五連），
    以 AI 視角編碼：0 空、1 自己、2 對手、3 棋盤外。
    w(s, a) = Π_{四個方向 d} card[棋形_d(s, a)]     （combine="product"）
            或 Σ_d card[棋形_d(s, a)]               （combine="sum"）
    P(a | s) = w(s, a) / Σ_b w(s, b)

每一個盤面仍然有自己的一副「位子牌組」（每個空位的牌數 w(s, a)），
只是牌數由共用的棋形牌拼出來，所以在 A 局學到的經驗可以用在 B 局的相同棋形上。

獎懲：本局 AI 每一手所用到的四張棋形牌，依結果一起增減。

對稱去重（use_symmetry=True）：
    - 一條線正著讀與反著讀視為同一棋形（鏡像）；
    - 四個方向共用同一套棋形牌（旋轉 90° 會把橫線變直線、正斜變反斜）。
    use_symmetry=False 時，棋形前面加上方向編號、且不合併正反，用來做對照實驗。

對手記憶（OpponentPatternMemory）：學習模式下一律記錄對手偏好的棋形；
opponent_model="preempt" 時，AI 會提高「對手很可能想下的位置」的牌數
（乘上 1 + preempt_strength × 偏好比率），也就是搶先佔住對手常下的點。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

from core.board import BLACK, SIZE, Board, to_rc
from core.symmetry import encode, relative_cells
from training.memory import OpponentPatternMemory, PatternCardMemory, action_to_str
from training.rewards import AI_LOSS, AI_WIN, DRAW, RewardConfig

from .base_agent import Agent, weighted_choice

FORMAT = "pattern-menace-weights"
FORMAT_VERSION = 1
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _build_lines(radius: int) -> list[list[list[int]]]:
    """LINES[a][d] = 位置 a 在方向 d 上的鄰格索引（-1 表示棋盤外），順序為 -radius..-1, +1..+radius。"""
    table = []
    for a in range(SIZE * SIZE):
        r, c = divmod(a, SIZE)
        per_dir = []
        for dr, dc in DIRECTIONS:
            cells = []
            for k in list(range(-radius, 0)) + list(range(1, radius + 1)):
                rr, cc = r + k * dr, c + k * dc
                cells.append(rr * SIZE + cc if 0 <= rr < SIZE and 0 <= cc < SIZE else -1)
            per_dir.append(cells)
        table.append(per_dir)
    return table


def _build_dir_windows() -> list[list[list[tuple[int, ...]]]]:
    """DIR_WINDOWS[a][d] = 方向 d 上所有通過位置 a、長度 5 的視窗。"""
    from core.rules import WINDOWS
    out = []
    for a in range(SIZE * SIZE):
        per_dir = []
        for dr, dc in DIRECTIONS:
            ws = []
            for w in WINDOWS:
                (r0, c0), (r1, c1) = divmod(w[0], SIZE), divmod(w[1], SIZE)
                if a in w and (r1 - r0, c1 - c0) == (dr, dc):
                    ws.append(w)
            per_dir.append(ws)
        out.append(per_dir)
    return out


DIR_WINDOWS = _build_dir_windows()


def window_keys(cells: list[int], a: int, color: int) -> list[str]:
    """位置 a 四個方向的「視窗計數」棋形（以 color 視角）。

    每個方向取通過 a 的所有長度 5 視窗：
        攻 = 不含對手子的視窗中，自己子數的最大值（-1 表示此方向已無法連五）
        攻n = 達到最大值的視窗數（1 或 2 以上；用來區分「活」與「死」）
        守 / 守n = 同理，看對手子數（不含自己子的視窗）
    編碼為 "攻,攻n|守,守n"，例如 "4,1|0,2"。四個方向共用同一套牌（等同旋轉對稱去重）。
    """
    keys = []
    for ws in DIR_WINDOWS[a]:
        om, on, dm, dn = -1, 0, -1, 0
        for w in ws:
            own = opp = 0
            for j in w:
                v = cells[j] * color
                if v > 0:
                    own += 1
                elif v < 0:
                    opp += 1
            if opp == 0:
                if own > om:
                    om, on = own, 1
                elif own == om:
                    on += 1
            if own == 0:
                if opp > dm:
                    dm, dn = opp, 1
                elif opp == dm:
                    dn += 1
        keys.append(f"{om},{min(on, 2)}|{dm},{min(dn, 2)}")
    return keys


_LINES_CACHE: dict[int, list[list[list[int]]]] = {}


def lines_for(radius: int) -> list[list[list[int]]]:
    if radius not in _LINES_CACHE:
        _LINES_CACHE[radius] = _build_lines(radius)
    return _LINES_CACHE[radius]


def pattern_keys(cells: list[int], a: int, color: int, radius: int = 4, use_symmetry: bool = True) -> list[str]:
    """位置 a 四個方向的棋形編碼（以 color 的視角）。"""
    keys = []
    for d, idxs in enumerate(lines_for(radius)[a]):
        chars = []
        for j in idxs:
            if j < 0:
                chars.append("3")
            else:
                v = cells[j] * color
                chars.append("0" if v == 0 else ("1" if v > 0 else "2"))
        s = "".join(chars)
        if use_symmetry:
            rs = s[::-1]
            keys.append(s if s <= rs else rs)
        else:
            keys.append(f"{d}:{s}")
    return keys


@dataclass
class PatternMenaceConfig:
    reward: RewardConfig = field(default_factory=RewardConfig)
    combine: str = "product"          # "product" 或 "sum"
    features: str = "line"            # "line"：左右各 radius 格的完整排列；"window"：視窗計數（見 window_keys）
    radius: int = 4
    use_symmetry: bool = True
    opponent_model: str = "off"       # "off" 或 "preempt"
    preempt_strength: float = 4.0
    version: str = "pattern-menace-1.0"

    def to_dict(self) -> dict:
        return {"reward": self.reward.to_dict(), "combine": self.combine, "features": self.features,
                "radius": self.radius,
                "use_symmetry": self.use_symmetry, "opponent_model": self.opponent_model,
                "preempt_strength": self.preempt_strength, "version": self.version}

    @classmethod
    def from_dict(cls, d: dict) -> "PatternMenaceConfig":
        cfg = cls(reward=RewardConfig.from_dict(d.get("reward", {})))
        for k in ("combine", "features", "radius", "use_symmetry", "opponent_model", "preempt_strength", "version"):
            if k in d:
                setattr(cfg, k, d[k])
        if cfg.features not in ("line", "window"):
            raise ValueError("features 必須是 'line' 或 'window'")
        if cfg.combine not in ("product", "sum"):
            raise ValueError("combine 必須是 'product' 或 'sum'")
        if cfg.opponent_model not in ("off", "preempt"):
            raise ValueError("opponent_model 必須是 'off' 或 'preempt'")
        return cfg


class PatternMenaceAgent(Agent):
    kind = "pattern_menace"

    def __init__(self, config: PatternMenaceConfig | None = None, name: str | None = None,
                 seed: int | None = None, learning: bool = True):
        super().__init__(name or "pattern-menace", seed)
        self.config = config or PatternMenaceConfig()
        r = self.config.reward
        self.memory = PatternCardMemory(r.initial_weight, r.min_weight, r.max_weight)
        self.opponent_memory = OpponentPatternMemory()
        self.learning = learning
        self.games_trained = 0
        self.loss_length_avg: float | None = None  # 輸棋平均手數（指數移動平均，存活獎勵用）
        self.steps: list[dict] = []
        self.color = BLACK

    @property
    def version(self) -> str:  # type: ignore[override]
        return self.config.version

    def params(self) -> dict:
        return {**self.config.to_dict(), "learning": self.learning, "games_trained": self.games_trained}

    def start_game(self, color: int, seed: int | None = None) -> None:
        super().start_game(color, seed)
        self.steps = []

    def _keys(self, cells: list[int], a: int, color: int) -> list[str]:
        if self.config.features == "window":
            return window_keys(cells, a, color)
        return pattern_keys(cells, a, color, self.config.radius, self.config.use_symmetry)

    def _combine(self, values: list[int]) -> float:
        return math.prod(values) if self.config.combine == "product" else sum(values)

    def deck(self, board: Board, color: int) -> tuple[list[int], list[list[str]], list[float]]:
        """回傳 (合法位置, 各位置四個棋形, 各位置牌數 w(s,a))。"""
        legal = board.legal_moves()
        keys = [self._keys(board.cells, a, color) for a in legal]
        weights = [self._combine([self.memory.get(k) for k in ks]) for ks in keys]
        return legal, keys, weights

    def select_move(self, board: Board, color: int) -> int:
        legal, keys, weights = self.deck(board, color)
        if not legal:
            raise ValueError("沒有合法落子位置")
        reset = False
        if sum(weights) <= 0:
            # 所有位置牌數皆為 0（min_weight=0 時可能發生）：暫時均勻選擇，不改動牌組
            weights = [1.0] * len(legal)
            reset = True
        effective = list(weights)
        if self.config.opponent_model == "preempt" and self.opponent_memory.observations:
            beta = self.config.preempt_strength
            for i, a in enumerate(legal):
                pref = self.opponent_memory.preference(self._keys(board.cells, a, -color))
                effective[i] *= 1.0 + beta * pref
        choice = weighted_choice(self.rng, effective)
        a = legal[choice]
        total = sum(weights)
        self.steps.append({
            "move_no": board.stone_count() + 1,
            "state": encode(relative_cells(board.cells, color)),
            "action": list(to_rc(a)),
            "patterns": keys[choice],
            "cards_before": [self.memory.get(k) for k in keys[choice]],
            "weight_before": weights[choice],
            "deck_total": total,
            "prob": round(effective[choice] / sum(effective), 6),
            "deck_size": len(legal),
            **({"uniform_fallback": True} if reset else {}),
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
            deltas = self._deltas(result)
            for step, d in zip(self.steps, deltas):
                for k in step["patterns"]:
                    self.memory.adjust(k, d)
                step["delta"] = d
            # 同一張棋形牌可能在本局被用到多次（例如四個方向棋形相同），所以等全部更新完再記錄最終牌數
            for step in self.steps:
                after = [self.memory.get(k) for k in step["patterns"]]
                step["cards_after"] = after
                step["weight_after"] = self._combine(after)
            self.games_trained += 1
        return {"result": result, "steps": self.steps}

    def _deltas(self, result: str) -> list[int]:
        """依獎懲設定計算本局各步的牌數增減；同時更新輸棋平均手數。"""
        game_length = self.steps[-1]["move_no"] + (1 if result == AI_LOSS else 0) if self.steps else 0
        deltas = self.config.reward.deltas(result, len(self.steps), game_length, self.loss_length_avg)
        if result == AI_LOSS:
            avg = self.loss_length_avg
            self.loss_length_avg = game_length if avg is None else 0.99 * avg + 0.01 * game_length
        return deltas

    def move_probabilities(self, board: Board, color: int) -> dict[int, float]:
        legal, _, w = self.deck(board, color)
        total = sum(w) or 1
        return {a: wi / total for a, wi in zip(legal, w)}

    # ---- 保存 / 載入 ------------------------------------------------------
    def to_dict(self) -> dict:
        return {"format": FORMAT, "format_version": FORMAT_VERSION, "config": self.config.to_dict(),
                "games_trained": self.games_trained, "loss_length_avg": self.loss_length_avg,
                "memory": self.memory.to_dict(),
                "opponent_memory": self.opponent_memory.to_dict()}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)

    @classmethod
    def from_dict(cls, data: dict, learning: bool = True, seed: int | None = None,
                  name: str | None = None) -> "PatternMenaceAgent":
        if data.get("format") != FORMAT:
            raise ValueError("不是棋形位子牌權重檔")
        agent = cls(PatternMenaceConfig.from_dict(data["config"]), name=name, seed=seed, learning=learning)
        agent.memory = PatternCardMemory.from_dict(data["memory"])
        agent.opponent_memory = OpponentPatternMemory.from_dict(data.get("opponent_memory", {}))
        agent.games_trained = data.get("games_trained", 0)
        agent.loss_length_avg = data.get("loss_length_avg")
        return agent

    @classmethod
    def load(cls, path: str, learning: bool = True, seed: int | None = None,
             name: str | None = None) -> "PatternMenaceAgent":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f), learning=learning, seed=seed, name=name)

    def twin(self, name: str | None = None) -> "PatternMenaceAgent":
        """自我對弈用：另一個「座位」，與本體共用同一副棋形牌與對手記憶。

        兩個座位各自記錄本局的選擇，終局時各自依自己的勝負更新同一副牌。
        """
        other = PatternMenaceAgent(self.config, name=name or f"{self.name}-self", learning=self.learning)
        other.memory = self.memory
        other.opponent_memory = self.opponent_memory
        other.kind = "self_play"
        return other

    def clone(self, learning: bool | None = None) -> "PatternMenaceAgent":
        return PatternMenaceAgent.from_dict(json.loads(json.dumps(self.to_dict())),
                                            learning=self.learning if learning is None else learning,
                                            name=self.name)


__all__ = ["PatternMenaceAgent", "PatternMenaceConfig", "pattern_keys", "action_to_str"]
