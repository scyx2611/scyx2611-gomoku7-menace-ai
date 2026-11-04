"""第五種模式：位子牌＋神經網路（只靠規則）。

與第四種模式（rule_only_card_agent.py）的差別只有「誰來估盤面」：
第四種用小區塊牌加總，這裡用一個小型卷積神經網路，讓 AI 自己組合遠距離的關係。

只靠規則
    網路只看「下完之後的盤面」三張 7×7 平面：剛落子一方的子、對手的子、全 1（讓網路分得出棋盤邊緣）。
    沒有「線」「子數」「攻」「守」的概念；選位置時也不檢查「下這裡會不會連五」。

位子牌組
    每個空位 a 的牌數 = 10 × e^(β·V(下在 a 之後的盤面))，V ∈ (−1, 1) 是網路估的勝算，依牌數比例抽籤。
    β 固定，所以永遠保留探索。

獎懲（增、減位子牌數量）
    終局後由最後一手往回算目標：最後一手 = 勝 +1 / 敗 −1 / 和 0，
    其餘手 = (1−λ) × 自己下一手盤面的估值 + λ × 再往後的目標（TD(λ)）；
    以均方誤差訓練網路，讓估值靠近目標。贏棋的盤面估值上升，同樣模樣的位置牌數變多；輸棋則變少。
    旋轉／鏡像：訓練樣本隨機套用八種轉換之一（只用到棋盤的幾何性質）。

需要 PyTorch。沒有安裝時，匯入本模組會丟出 ImportError，其他模式不受影響。
"""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.board import BLACK, COLOR_NAME, SIZE, WHITE, Board, to_rc
from core.rules import is_winning_move
from core.symmetry import INV_PERM, encode, relative_cells
from training.memory import OpponentMemory
from training.rewards import AI_LOSS, AI_WIN, DRAW

from .base_agent import Agent, weighted_choice

FORMAT = "neural-card-weights"
FORMAT_VERSION = 1
N = SIZE * SIZE
INV8 = np.array(INV_PERM)


class ValueNet(nn.Module):
    """小型殘差卷積網路：3×7×7 → 勝算估值。"""

    def __init__(self, channels: int = 16, blocks: int = 3):
        super().__init__()
        self.inp = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.ModuleList([nn.Conv2d(channels, channels, 3, padding=1) for _ in range(blocks)])
        self.v1 = nn.Linear(channels * N, 64)
        self.v2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.inp(x))
        for conv in self.body:
            h = F.relu(h + conv(h))
        return torch.tanh(self.v2(F.relu(self.v1(h.flatten(1))))).squeeze(1)


def planes(rel: np.ndarray) -> torch.Tensor:
    """rel：(B, 49)，1 = 剛落子的一方、−1 = 對手、0 = 空。"""
    own = (rel == 1).astype(np.float32)
    opp = (rel == -1).astype(np.float32)
    return torch.from_numpy(np.stack([own, opp, np.ones_like(own)], 1).reshape(-1, 3, SIZE, SIZE))


def afterstates(cells: np.ndarray, color: int) -> tuple[np.ndarray, np.ndarray]:
    """一個盤面所有空位「下完之後」的相對盤面。"""
    legal = np.flatnonzero(cells == 0)
    rel = np.repeat((cells * color)[None], len(legal), 0)
    rel[np.arange(len(legal)), legal] = 1
    return legal, rel


@dataclass
class NeuralCardConfig:
    channels: int = 16
    blocks: int = 3
    beta: float = 10.0
    lam: float = 0.7
    lr: float = 1e-3
    batch_games: int = 64
    initial_weight: float = 10.0
    version: str = "neural-card-1.0"

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "NeuralCardConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class NetMemory:
    """網路與最佳化器。to_dict 只回傳參數的雜湊值，用來檢查評估前後網路是否被改動。"""

    def __init__(self, config: NeuralCardConfig, seed: int = 0):
        with torch.random.fork_rng():          # 初始權重由 seed 決定，不影響全域亂數狀態
            torch.manual_seed(seed)
            self.net = ValueNet(config.channels, config.blocks)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=config.lr)

    def __len__(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def to_dict(self) -> dict:
        h = hashlib.sha256()
        for p in self.net.state_dict().values():
            h.update(p.detach().cpu().numpy().tobytes())
        return {"sha256": h.hexdigest(), "parameters": len(self)}


class NeuralCardAgent(Agent):
    kind = "neural_card"

    def __init__(self, config: NeuralCardConfig | None = None, name: str | None = None,
                 seed: int | None = None, learning: bool = True):
        super().__init__(name or "neural-card", seed)
        self.config = config or NeuralCardConfig()
        self.memory = NetMemory(self.config, 0 if seed is None else seed)
        self.opponent_memory = OpponentMemory()   # 只為介面一致；此模式不記錄對手
        self.learning = learning
        self.games_trained = 0
        self.steps: list[dict] = []
        self._train: list[tuple[np.ndarray, float]] = []
        self.color = BLACK

    @property
    def net(self) -> ValueNet:
        return self.memory.net

    @property
    def version(self) -> str:  # type: ignore[override]
        return self.config.version

    def params(self) -> dict:
        return {**self.config.to_dict(), "learning": self.learning, "games_trained": self.games_trained}

    # ---- 估值與位子牌 -------------------------------------------------------
    def values(self, rel: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.net(planes(rel)).numpy().astype(np.float64)

    def _probs(self, v: np.ndarray) -> np.ndarray:
        x = self.config.beta * (v - v.max())
        p = np.exp(x)
        return p / p.sum()

    def card(self, v: float) -> float:
        return float(self.config.initial_weight * np.exp(self.config.beta * v))

    def move_probabilities(self, board: Board, color: int) -> dict[int, float]:
        legal, rel = afterstates(np.array(board.cells, dtype=np.int8), color)
        return dict(zip(legal.tolist(), self._probs(self.values(rel)).tolist()))

    # ---- 一局一局對弈（介面、評估、介面中的「開始訓練」）----------------------
    def start_game(self, color: int, seed: int | None = None) -> None:
        super().start_game(color, seed)
        self.steps = []
        self._train = []

    def select_move(self, board: Board, color: int) -> int:
        legal, rel = afterstates(np.array(board.cells, dtype=np.int8), color)
        v = self.values(rel)
        probs = self._probs(v)
        i = weighted_choice(self.rng, probs.tolist())
        if self.learning:
            self._train.append((rel[i].copy(), float(v[i])))
        self.steps.append({"move_no": board.stone_count() + 1, "state": encode(relative_cells(board.cells, color)),
                           "action": list(to_rc(int(legal[i]))), "value_before": round(float(v[i]), 5),
                           "card_before": round(self.card(v[i]), 4), "prob": round(float(probs[i]), 6),
                           "deck_size": len(legal)})
        return int(legal[i])

    def end_game(self, winner: int | None, my_color: int) -> dict:
        result = DRAW if winner is None else (AI_WIN if winner == my_color else AI_LOSS)
        if self.learning and self._train:
            outcome = {AI_WIN: 1.0, AI_LOSS: -1.0, DRAW: 0.0}[result]
            targets = self._targets([v for _, v in self._train], outcome)
            self._fit(np.stack([r for r, _ in self._train]), np.array(targets), np.random.default_rng())
            self.games_trained += 1
            for st, t in zip(self.steps, targets):
                st["target"] = round(t, 5)
        return {"result": result, "steps": self.steps}

    # ---- 學習 -------------------------------------------------------------
    def _targets(self, values: list[float], outcome: float) -> list[float]:
        """TD(λ) 目標：由最後一手往回算。"""
        lam = self.config.lam
        out = [0.0] * len(values)
        g = outcome
        for t in range(len(values) - 1, -1, -1):
            if t < len(values) - 1:
                g = (1 - lam) * values[t + 1] + lam * g
            out[t] = g
        return out

    def _fit(self, rel: np.ndarray, targets: np.ndarray, rng: np.random.Generator) -> float:
        tr = rng.integers(0, 8, len(rel))
        x = planes(np.take_along_axis(rel, INV8[tr], axis=1))   # 每個樣本隨機套一種旋轉/鏡像
        loss = F.mse_loss(self.net(x), torch.tensor(targets, dtype=torch.float32))
        self.memory.opt.zero_grad()
        loss.backward()
        self.memory.opt.step()
        return float(loss.detach())

    def train_games(self, n_games: int, *, base_seed: int, start_index: int, random_frac: float = 0.1,
                    logger=None, experiment: str = "train", include_learning: bool = False, tally=None) -> list[dict]:
        """批次自我對弈：同時下 batch_games 局，每批結束後學習一次。回傳每局統計列。

        第 k 局（全域局號 idx）：以 random_frac 的機率是「AI vs 隨機」，否則是自我對弈；
        紀錄中的 AI 座位：自我對弈為奇數局黑、偶數局白，與隨機對下則是非隨機的一方。
        """
        rows: list[dict] = []
        done = 0
        while done < n_games:
            b = min(self.config.batch_games, n_games - done)
            idx0 = start_index + done
            rng = np.random.default_rng([base_seed, idx0])
            rows += self._batch(b, idx0, rng, random_frac, logger, experiment, include_learning, tally, base_seed)
            done += b
        return rows

    def _batch(self, B, idx0, rng, random_frac, logger, experiment, include_learning, tally, base_seed):
        cells = np.zeros((B, N), dtype=np.int8)
        to_move = np.ones(B, dtype=np.int8)
        done = np.zeros(B, bool)
        winner = np.zeros(B, dtype=np.int8)
        rand_side = np.where(rng.random(B) < random_frac, rng.choice([1, -1], B), 0).astype(np.int8)
        moves: list[list[tuple[int, int]]] = [[] for _ in range(B)]
        learn: list[list[dict]] = [[] for _ in range(B)]
        while not done.all():
            act = np.nonzero(~done)[0]
            groups = [afterstates(cells[g], int(to_move[g])) for g in act]
            v_all = self.values(np.concatenate([r for _, r in groups]))
            pos = 0
            for (legal, rel), g in zip(groups, act):
                v = v_all[pos:pos + len(legal)]
                pos += len(legal)
                color = int(to_move[g])
                if rand_side[g] == color:
                    i = int(rng.integers(len(legal)))
                else:
                    p = self._probs(v)
                    i = int(rng.choice(len(legal), p=p))
                    learn[g].append({"color": color, "rel": rel[i].copy(), "v": float(v[i]), "prob": float(p[i]),
                                     "move_no": len(moves[g]) + 1, "action": int(legal[i]),
                                     "state": encode((cells[g] * color).tolist()), "deck": len(legal)})
                a = int(legal[i])
                cells[g, a] = color
                moves[g].append((a, color))
                r, c = divmod(a, SIZE)
                if is_winning_move(Board(cells[g].tolist()), r, c, color):
                    done[g] = True
                    winner[g] = color
                elif not (cells[g] == 0).any():
                    done[g] = True
                to_move[g] = -color
        # 學習：每局、每位（非隨機的）玩家各自算 TD(λ) 目標
        X, Y, refs = [], [], []
        for g in range(B):
            for color in (BLACK, WHITE):
                st = [s for s in learn[g] if s["color"] == color]
                if not st:
                    continue
                outcome = 0.0 if winner[g] == 0 else (1.0 if winner[g] == color else -1.0)
                for s, t in zip(st, self._targets([s["v"] for s in st], outcome)):
                    s["target"] = t
                    X.append(s["rel"])
                    Y.append(t)
                    refs.append(s)
        if X:
            self._fit(np.stack(X), np.array(Y), rng)
        if include_learning and X:
            after = self.values(np.stack(X))
            for s, va in zip(refs, after):
                s["value_after"] = float(va)
        self.games_trained += B
        # 紀錄與統計列
        rows = []
        for g in range(B):
            idx = idx0 + g
            self_play = rand_side[g] == 0
            ai_color = (BLACK if idx % 2 == 1 else WHITE) if self_play else int(-rand_side[g])
            w = int(winner[g])
            result = DRAW if w == 0 else (AI_WIN if w == ai_color else AI_LOSS)
            game_id = f"{experiment}-train-{idx:06d}"
            if logger is not None:
                rec = {
                    "game_id": game_id, "experiment": experiment, "phase": "train",
                    "ai": {"type": self.kind, "version": self.version, "name": self.name},
                    "opponent": {"type": "self_play" if self_play else "random"},
                    "seed": [base_seed, idx0], "ai_color": COLOR_NAME[ai_color], "ai_first": ai_color == BLACK,
                    "moves": [{"n": k + 1, "player": COLOR_NAME[col], "row": a // SIZE, "col": a % SIZE,
                               "by": "ai" if (self_play or col == ai_color) else "opponent"}
                              for k, (a, col) in enumerate(moves[g])],
                    "move_count": len(moves[g]), "winner": COLOR_NAME.get(w) if w else None, "result": result,
                    "reward_params": {k: self.config.to_dict()[k] for k in ("beta", "lam", "lr", "initial_weight")},
                }
                if include_learning:
                    rec["learning"] = [{
                        "player": COLOR_NAME[s["color"]], "move_no": s["move_no"], "state": s["state"],
                        "action": list(to_rc(s["action"])), "value_before": round(s["v"], 5),
                        "card_before": round(self.card(s["v"]), 4), "prob": round(s["prob"], 6),
                        "deck_size": s["deck"], "target": round(s["target"], 5),
                        "value_after": round(s["value_after"], 5), "card_after": round(self.card(s["value_after"]), 4),
                    } for s in learn[g]]
                logger.write(rec)
            row = {"game_no": idx, "game_id": game_id, "result": result, "move_count": len(moves[g]),
                   "ai_color": COLOR_NAME[ai_color], "opponent_used": "self-play" if self_play else "random"}
            if tally is not None:
                row.update(tally.add(result))
            row["learned_entries"] = len(self.memory)
            rows.append(row)
        return rows

    # ---- 自我對弈 / 保存 ---------------------------------------------------
    def twin(self, name: str | None = None) -> "NeuralCardAgent":
        other = NeuralCardAgent(self.config, name=name or f"{self.name}-self", learning=self.learning)
        other.memory = self.memory
        other.kind = "self_play"
        return other

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        torch.save({"format": FORMAT, "format_version": FORMAT_VERSION, "config": self.config.to_dict(),
                    "games_trained": self.games_trained, "state_dict": self.net.state_dict()}, buf)
        return buf.getvalue()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(self.to_bytes())
        os.replace(tmp, path)

    @classmethod
    def from_bytes(cls, data: bytes, learning: bool = True, seed: int | None = None,
                   name: str | None = None) -> "NeuralCardAgent":
        d = torch.load(io.BytesIO(data), weights_only=True)
        if d.get("format") != FORMAT:
            raise ValueError("不是神經網路位子牌權重檔")
        agent = cls(NeuralCardConfig.from_dict(d["config"]), name=name, seed=seed, learning=learning)
        agent.net.load_state_dict(d["state_dict"])
        agent.games_trained = d.get("games_trained", 0)
        return agent

    @classmethod
    def load(cls, path: str, learning: bool = True, seed: int | None = None,
             name: str | None = None) -> "NeuralCardAgent":
        with open(path, "rb") as f:
            return cls.from_bytes(f.read(), learning=learning, seed=seed, name=name)

    def clone(self, learning: bool | None = None) -> "NeuralCardAgent":
        return NeuralCardAgent.from_bytes(self.to_bytes(), learning=self.learning if learning is None else learning,
                                          name=self.name)
