"""第四種模式：只靠規則的位子牌 AI。

除了五子棋規則（哪裡能下、誰贏、和局），程式不預先判斷任何東西：
沒有「線」「子數」「進攻」「防守」的概念，也不在選位置時檢查「下這裡會不會連五」。

位子牌組（老師的「各階段局面的位子牌組」）
    每到一個局面，每個空位 a 的牌數 = 10 × e^(β·V(下在 a 之後的盤面))，依牌數比例抽籤。
    V(盤面) = tanh(所有小區塊牌值的總和 ÷ 8)，以剛落子的一方為「自己」。
    小區塊：棋盤上所有 3×3、2×4、4×2 的長方形，每格只記原始內容（自己 / 對手 / 空），
    每一種內容組合是一張牌。位置對稱的小區塊共用同一套牌，而且每次估值都把盤面的
    八種旋轉／鏡像各估一次取平均，所以互為旋轉、鏡像的局面一定得到相同的牌數。

獎懲（增、減位子牌數量）
    終局後由最後一手往回調整本局下完後的每個盤面：
        最後一手的目標 = 勝 +1 / 敗 −1 / 和 0（由規則判定）
        其餘手的目標   = (1−λ) × 自己下一手盤面的估值 + λ × 再往後的目標（TD(λ)）
        小區塊牌值 += α × (目標 − 目前估值) × (1 − 估值²) ÷ 小區塊數
    贏棋的盤面裡出現過的小區塊牌值上升，下次同樣模樣的位置牌數變多；輸棋則變少。
"""

from __future__ import annotations

import gzip
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from core.board import BLACK, SIZE, Board, to_rc
from core.symmetry import INV_PERM, PERM, encode, relative_cells
from training.memory import OpponentMemory
from training.rewards import AI_LOSS, AI_WIN, DRAW

from .base_agent import Agent, weighted_choice

FORMAT = "rule-only-card-weights"
FORMAT_VERSION = 1
N = SIZE * SIZE
_M = SIZE - 1
_SYM = (
    lambda r, c: (r, c), lambda r, c: (c, _M - r), lambda r, c: (_M - r, _M - c), lambda r, c: (_M - c, r),
    lambda r, c: (r, _M - c), lambda r, c: (_M - r, c), lambda r, c: (c, r), lambda r, c: (_M - c, _M - r),
)
PERM8 = np.array(PERM)
INV8 = np.array(INV_PERM)


class WindowTables:
    """把棋盤上所有指定形狀的小區塊依八種對稱分類，並預先算好查表用的索引。"""

    def __init__(self, shapes: tuple[tuple[int, int], ...]):
        windows = []
        for h, w in shapes:
            for r in range(SIZE - h + 1):
                for c in range(SIZE - w + 1):
                    windows.append(tuple((r + i) * SIZE + (c + j) for i in range(h) for j in range(w)))
        classes: dict[tuple[int, ...], int] = {}
        class_of, cells, powk = [], [], []
        for win in windows:
            best = None
            for f in _SYM:
                mapped = {cell: f(*divmod(cell, SIZE)) for cell in win}
                key = tuple(sorted(r * SIZE + c for r, c in mapped.values()))
                if best is None or key < best[0]:
                    best = (key, mapped)
            key, mapped = best
            class_of.append(classes.setdefault(key, len(classes)))
            order = {cell: i for i, cell in enumerate(key)}
            cells.append(list(win))
            powk.append([3 ** order[r * SIZE + c] for r, c in (mapped[cell] for cell in win)])
        self.n_windows = len(windows)
        self.n_classes = len(classes)
        sizes = [len(k) for k in classes]
        offset = np.cumsum([0] + [3 ** n for n in sizes])
        self.n_features = int(offset[-1])
        maxk = max(len(c) for c in cells)
        self.cells = np.array([c + [N] * (maxk - len(c)) for c in cells])     # N = 補位的虛擬空格
        self.powk = np.array([p + [0] * (maxk - len(p)) for p in powk])
        self.base = offset[np.array(class_of)]
        cw = [[] for _ in range(N)]
        cp = [[] for _ in range(N)]
        for w, (cl, pl) in enumerate(zip(cells, powk)):
            for cell, p in zip(cl, pl):
                cw[cell].append(w)
                cp[cell].append(p)
        maxc = max(len(x) for x in cw)
        self.cell_windows = np.full((N, maxc), self.n_windows)
        self.cell_powers = np.zeros((N, maxc), dtype=np.int64)
        for a in range(N):
            self.cell_windows[a, :len(cw[a])] = cw[a]
            self.cell_powers[a, :len(cp[a])] = cp[a]


_TABLES: dict[tuple, WindowTables] = {}


def tables_for(shapes) -> WindowTables:
    key = tuple(tuple(s) for s in shapes)
    if key not in _TABLES:
        _TABLES[key] = WindowTables(key)
    return _TABLES[key]


@dataclass
class RuleOnlyConfig:
    shapes: list = field(default_factory=lambda: [[3, 3], [2, 4], [4, 2]])
    alpha: float = 0.1      # 學習率
    lam: float = 0.7        # TD(λ)：0 = 只看下一手估值，1 = 只看終局結果
    beta: float = 10.0      # 牌數對估值的敏感度：越大越常選估值最高的位置
    initial_weight: float = 10.0   # 估值 0 的位置有幾張牌（只影響顯示）
    # 是否記錄對手落子（只做統計，不影響選位置）。自我對弈 30 萬局會累積數 GB，預設關閉。
    record_opponent: bool = False
    version: str = "rule-only-1.0"

    def to_dict(self) -> dict:
        return {"shapes": [list(s) for s in self.shapes], "alpha": self.alpha, "lam": self.lam, "beta": self.beta,
                "initial_weight": self.initial_weight, "record_opponent": self.record_opponent,
                "version": self.version}

    @classmethod
    def from_dict(cls, d: dict) -> "RuleOnlyConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class WeightMemory:
    """小區塊牌值（numpy 陣列）。提供 to_dict 讓評估時可比對學習資料是否被改動。"""

    def __init__(self, n_features: int):
        self.w = np.zeros(n_features + 1)   # 最後一格固定為 0（補位用）

    def __len__(self) -> int:
        return int(np.count_nonzero(self.w))

    def to_dict(self) -> dict:
        nz = np.flatnonzero(self.w)
        return {"size": len(self.w), "index": nz.tolist(), "value": self.w[nz].tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "WeightMemory":
        m = cls(d["size"] - 1)
        m.w[np.array(d["index"], dtype=np.int64)] = np.array(d["value"])
        return m


class RuleOnlyCardAgent(Agent):
    kind = "rule_only_card"

    def __init__(self, config: RuleOnlyConfig | None = None, name: str | None = None,
                 seed: int | None = None, learning: bool = True):
        super().__init__(name or "rule-only-card", seed)
        self.config = config or RuleOnlyConfig()
        self.t = tables_for(self.config.shapes)
        self.memory = WeightMemory(self.t.n_features)
        self.opponent_memory = OpponentMemory()
        self.learning = learning
        self.games_trained = 0
        self.record_after = False   # True 時，終局學習後重新計算每一手抽中的牌現在有幾張（寫進棋譜）
        self.steps: list[dict] = []
        self._train: list[tuple[np.ndarray, float]] = []
        self.color = BLACK

    @property
    def version(self) -> str:  # type: ignore[override]
        return self.config.version

    def params(self) -> dict:
        return {**self.config.to_dict(), "learning": self.learning, "games_trained": self.games_trained}

    def start_game(self, color: int, seed: int | None = None) -> None:
        super().start_game(color, seed)
        self.steps = []
        self._train = []

    # ---- 估值與位子牌 -------------------------------------------------------
    def after_values(self, board: Board, color: int):
        """回傳 (合法位置, 各位置下完後的估值, 供學習用的中間結果)。"""
        t, w = self.t, self.memory.w
        legal = np.array(board.legal_moves())
        rel = np.asarray(board.cells) * color
        val = np.where(rel == 1, 1, np.where(rel == -1, 2, 0))
        z = np.zeros(len(legal))
        per_t = []
        for k in range(8):
            vt = np.append(val[INV8[k]], 0)
            ids = t.base + (vt[t.cells] * t.powk).sum(axis=1)
            ids_pad = np.append(ids, t.n_features)
            lt = PERM8[k][legal]
            wsel = t.cell_windows[lt]
            old = ids_pad[wsel]
            new = np.where(wsel < t.n_windows, old + t.cell_powers[lt], t.n_features)
            z += w[ids].sum() + (w[new] - w[old]).sum(axis=1)
            per_t.append((ids_pad, wsel, new))
        return legal, np.tanh(z / 8), per_t

    def _probs(self, v: np.ndarray) -> np.ndarray:
        x = self.config.beta * (v - v.max())
        p = np.exp(x)
        return p / p.sum()

    def cards(self, v: np.ndarray) -> np.ndarray:
        return self.config.initial_weight * np.exp(self.config.beta * v)

    def move_probabilities(self, board: Board, color: int) -> dict[int, float]:
        legal, v, _ = self.after_values(board, color)
        return dict(zip(legal.tolist(), self._probs(v).tolist()))

    def select_move(self, board: Board, color: int) -> int:
        legal, v, per_t = self.after_values(board, color)
        probs = self._probs(v)
        i = weighted_choice(self.rng, probs.tolist())
        a = int(legal[i])
        if self.learning:
            feats = []
            for ids_pad, wsel, new in per_t:
                after = ids_pad.copy()
                m = wsel[i] < self.t.n_windows
                after[wsel[i][m]] = new[i][m]
                feats.append(after[:self.t.n_windows])
            self._train.append((np.concatenate(feats), float(v[i])))
        self.steps.append({
            "move_no": board.stone_count() + 1,
            "state": encode(relative_cells(board.cells, color)),
            "action": list(to_rc(a)),
            "value_before": round(float(v[i]), 5),
            "card_before": round(float(self.cards(v[i:i + 1])[0]), 4),
            "prob": round(float(probs[i]), 6),
            "deck_size": len(legal),
            "_cells": list(board.cells),
            "_color": color,
        })
        return a

    def observe_move(self, board_before: Board, move: int, color: int) -> None:
        if color == self.color or not self.learning or not self.config.record_opponent:
            return
        from core.symmetry import canonical
        key, tr = canonical(relative_cells(board_before.cells, self.color))
        self.opponent_memory.record(key, PERM[tr][move], False)

    def end_game(self, winner: int | None, my_color: int) -> dict:
        result = DRAW if winner is None else (AI_WIN if winner == my_color else AI_LOSS)
        if self.learning and self._train:
            c = self.config
            w = self.memory.w
            g = {AI_WIN: 1.0, AI_LOSS: -1.0, DRAW: 0.0}[result]
            nw = self.t.n_windows
            for k in range(len(self._train) - 1, -1, -1):
                ids, v_old = self._train[k]
                if k < len(self._train) - 1:
                    g = (1 - c.lam) * self._train[k + 1][1] + c.lam * g
                cur = math.tanh(w[ids].sum() / 8)
                np.add.at(w, ids, c.alpha / nw * (g - cur) * (1 - cur * cur))
                self.steps[k]["target"] = round(g, 5)
            self.games_trained += 1
            if self.record_after:
                # 學習後再看一次同一個局面：抽中的那張位子牌現在有幾張
                for st in self.steps:
                    legal, v, _ = self.after_values(Board(st["_cells"]), st["_color"])
                    j = int(np.flatnonzero(legal == st["action"][0] * SIZE + st["action"][1])[0])
                    st["value_after"] = round(float(v[j]), 5)
                    st["card_after"] = round(float(self.cards(v[j:j + 1])[0]), 4)
        for st in self.steps:
            st.pop("_cells", None)
            st.pop("_color", None)
        return {"result": result, "steps": self.steps}

    # ---- 自我對弈 / 保存 ---------------------------------------------------
    def twin(self, name: str | None = None) -> "RuleOnlyCardAgent":
        other = RuleOnlyCardAgent(self.config, name=name or f"{self.name}-self", learning=self.learning)
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
        opener = gzip.open if path.endswith(".gz") else open
        with opener(tmp, "wt", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)

    @classmethod
    def from_dict(cls, data: dict, learning: bool = True, seed: int | None = None,
                  name: str | None = None) -> "RuleOnlyCardAgent":
        if data.get("format") != FORMAT:
            raise ValueError("不是只靠規則的位子牌權重檔")
        agent = cls(RuleOnlyConfig.from_dict(data["config"]), name=name, seed=seed, learning=learning)
        agent.memory = WeightMemory.from_dict(data["memory"])
        agent.opponent_memory = OpponentMemory.from_dict(data.get("opponent_memory", {}))
        agent.games_trained = data.get("games_trained", 0)
        return agent

    @classmethod
    def load(cls, path: str, learning: bool = True, seed: int | None = None,
             name: str | None = None) -> "RuleOnlyCardAgent":
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            return cls.from_dict(json.load(f), learning=learning, seed=seed, name=name)

    def clone(self, learning: bool | None = None) -> "RuleOnlyCardAgent":
        other = RuleOnlyCardAgent(self.config, name=self.name,
                                  learning=self.learning if learning is None else learning)
        other.memory.w = self.memory.w.copy()
        other.opponent_memory = OpponentMemory.from_dict(json.loads(json.dumps(self.opponent_memory.to_dict())))
        other.games_trained = self.games_trained
        return other
