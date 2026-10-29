"""MENACE 類型的「位子牌組」學習 AI。

選擇方式：
    P(a | s) = w(s, a) / Σ_b w(s, b)，b 為局面 s 的所有合法位置
    w(s, a) = 局面 s 中位置 a 的牌數。

每局結束後依 RewardConfig 回頭增減本局實際選過的 (局面, 位置) 牌數。

局面識別碼：相對於 AI 的盤面（自己=1、對手=2、空=0）編成 49 字元字串；
use_symmetry=True 時再取八種對稱中的標準形（core/symmetry.py），
並把位置換到標準座標儲存，落子時再轉回實際座標。

對手記憶（OpponentMemory）在學習模式下一律記錄；
opponent_model="avoid" 時，AI 會降低「下完後的局面，對手曾在該局面直接五連獲勝」
的位置的選擇機率（乘上 opponent_avoid_factor，不會變成 0，所以仍是合法可選）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from core.board import BLACK, WHITE, Board, to_rc
from core.rules import would_win
from core.symmetry import INV_PERM, PERM, encode, relative_cells
from training.memory import OpponentMemory, PositionCardMemory, action_to_str
from training.rewards import AI_LOSS, AI_WIN, DRAW, RewardConfig

from .base_agent import Agent, weighted_choice

FORMAT = "menace-weights"
FORMAT_VERSION = 1


@dataclass
class MenaceConfig:
    reward: RewardConfig = field(default_factory=RewardConfig)
    use_symmetry: bool = False
    opponent_model: str = "off"          # "off" 或 "avoid"
    opponent_avoid_factor: float = 0.1
    version: str = "menace-1.0"

    def to_dict(self) -> dict:
        return {
            "reward": self.reward.to_dict(),
            "use_symmetry": self.use_symmetry,
            "opponent_model": self.opponent_model,
            "opponent_avoid_factor": self.opponent_avoid_factor,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MenaceConfig":
        reward = RewardConfig.from_dict(d.get("reward", {}))
        cfg = cls(reward=reward)
        for k in ("use_symmetry", "opponent_model", "opponent_avoid_factor", "version"):
            if k in d:
                setattr(cfg, k, d[k])
        if cfg.opponent_model not in ("off", "avoid"):
            raise ValueError("opponent_model 必須是 'off' 或 'avoid'")
        return cfg


class MenaceAgent(Agent):
    kind = "menace"

    def __init__(self, config: MenaceConfig | None = None, name: str | None = None,
                 seed: int | None = None, learning: bool = True):
        super().__init__(name or "menace", seed)
        self.config = config or MenaceConfig()
        r = self.config.reward
        self.memory = PositionCardMemory(r.initial_weight, r.min_weight, r.max_weight)
        self.opponent_memory = OpponentMemory()
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

    # ---- 局面識別 ---------------------------------------------------------
    def _keys(self, rel: list[int]) -> list[str]:
        """可能的編碼字串（不使用對稱時只有原始方向）。"""
        s = encode(rel)
        if not self.config.use_symmetry:
            return [s]
        return ["".join(map(s.__getitem__, inv)) for inv in INV_PERM]

    @staticmethod
    def _best(keys: list[str]) -> tuple[str, int]:
        t = min(range(len(keys)), key=lambda i: (keys[i], i))
        return keys[t], t

    def state_key(self, cells: list[int], color: int) -> tuple[str, int]:
        """(局面識別碼, 轉換編號)；轉換編號用於座標換算。"""
        return self._best(self._keys(relative_cells(cells, color)))

    # ---- 對弈 -------------------------------------------------------------
    def start_game(self, color: int, seed: int | None = None) -> None:
        super().start_game(color, seed)
        self.steps = []

    def _avoid_flags(self, keys: list[str], legal: list[int]) -> list[bool]:
        """對每個候選位置判斷：下完後的局面，對手過去是否曾在此直接五連。"""
        flags = []
        for a in legal:
            nxt = []
            for t, k in enumerate(keys):
                p = PERM[t][a]
                nxt.append(k[:p] + "1" + k[p + 1:])
            key, _ = self._best(nxt)
            flags.append(self.opponent_memory.has_winning_reply(key))
        return flags

    def select_move(self, board: Board, color: int) -> int:
        keys = self._keys(relative_cells(board.cells, color))
        key, t = self._best(keys)
        legal = board.legal_moves()
        if not legal:
            raise ValueError("沒有合法落子位置")
        perm = PERM[t]
        legal_canon = [perm[a] for a in legal]
        weights = self.memory.deck(key, legal_canon)
        reset = False
        if sum(weights) <= 0:
            # 所有位置牌數歸零：學習模式下重設此局面，評估模式下僅暫時均勻選擇（不改權重）
            if self.learning:
                self.memory.reset_deck(key)
                weights = self.memory.deck(key, legal_canon)
            else:
                weights = [1] * len(legal)
            reset = True
        effective: list[float] = list(weights)
        avoided = 0
        if self.config.opponent_model == "avoid" and len(self.opponent_memory):
            flags = self._avoid_flags(keys, legal)
            if not all(flags):  # 全部都危險時就不調整
                for i, f in enumerate(flags):
                    if f:
                        effective[i] *= self.config.opponent_avoid_factor
                        avoided += 1
        choice = weighted_choice(self.rng, effective)
        if self.learning:
            self.memory.register(key)
        action = legal[choice]
        self.steps.append({
            "move_no": board.stone_count() + 1,
            "state": key,
            "transform": t,
            "action": list(to_rc(action)),
            "canon_action": action_to_str(legal_canon[choice]),
            "weight_before": weights[choice],
            "deck_total": sum(weights),
            "deck_size": len(legal),
            **({"avoided": avoided} if avoided else {}),
            **({"deck_reset": True} if reset else {}),
        })
        return action

    def observe_move(self, board_before: Board, move: int, color: int) -> None:
        if color == self.color or not self.learning:
            return
        key, t = self.state_key(board_before.cells, self.color)
        self.opponent_memory.record(key, PERM[t][move], would_win(board_before, move, color))

    def end_game(self, winner: int | None, my_color: int) -> dict:
        if winner is None:
            result = DRAW
        else:
            result = AI_WIN if winner == my_color else AI_LOSS
        if self.learning:
            deltas = self._deltas(result)
            for step, d in zip(self.steps, deltas):
                ca = step["canon_action"]
                r, c = ca.split(",")
                before, after = self.memory.adjust(step["state"], int(r) * 7 + int(c), d)
                step["delta"] = d
                step["weight_after"] = after
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

    # ---- 機率查詢（UI / 報告用）--------------------------------------------
    def move_probabilities(self, board: Board, color: int) -> dict[int, float]:
        key, t = self.state_key(board.cells, color)
        legal = board.legal_moves()
        w = self.memory.deck(key, [PERM[t][a] for a in legal])
        total = sum(w) or 1
        return {a: wi / total for a, wi in zip(legal, w)}

    # ---- 保存 / 載入 ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "config": self.config.to_dict(),
            "games_trained": self.games_trained,
            "loss_length_avg": self.loss_length_avg,
            "memory": self.memory.to_dict(),
            "opponent_memory": self.opponent_memory.to_dict(),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)

    @classmethod
    def from_dict(cls, data: dict, learning: bool = True, seed: int | None = None, name: str | None = None) -> "MenaceAgent":
        if data.get("format") != FORMAT:
            raise ValueError("不是 MENACE 權重檔")
        agent = cls(MenaceConfig.from_dict(data["config"]), name=name, seed=seed, learning=learning)
        agent.memory = PositionCardMemory.from_dict(data["memory"])
        agent.opponent_memory = OpponentMemory.from_dict(data.get("opponent_memory", {}))
        agent.games_trained = data.get("games_trained", 0)
        agent.loss_length_avg = data.get("loss_length_avg")
        return agent

    @classmethod
    def load(cls, path: str, learning: bool = True, seed: int | None = None, name: str | None = None) -> "MenaceAgent":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f), learning=learning, seed=seed, name=name)

    def clone(self, learning: bool | None = None) -> "MenaceAgent":
        """深拷貝（評估時使用凍結的副本，保證不影響訓練中的權重）。"""
        return MenaceAgent.from_dict(json.loads(json.dumps(self.to_dict())),
                                     learning=self.learning if learning is None else learning, name=self.name)


__all__ = ["MenaceAgent", "MenaceConfig", "WHITE"]
