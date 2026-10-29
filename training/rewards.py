"""獎懲規則：依棋局結果計算 AI 每一個選擇的「位子牌」增減量。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

AI_WIN = "ai_win"
AI_LOSS = "ai_loss"
DRAW = "draw"


@dataclass
class RewardConfig:
    win_reward: int = 3        # 勝利：本局每個選擇 +3 張牌
    loss_penalty: int = 1      # 失敗：本局每個選擇 -1 張牌
    draw_reward: int = 1       # 平局：+1（設 0 即不變）
    initial_weight: int = 10   # 新局面每個合法位置的初始牌數
    min_weight: int = 1        # 牌數下限（設 0 允許把某位置完全移除）
    # 失敗時 AI「最後一手」（下完後對手直接五連）的懲罰；None 表示與 loss_penalty 相同
    final_loss_penalty: int | None = None
    # 存活獎勵（相對獎懲）：輸棋時若本局手數比「過去輸棋的平均手數」多，改給 +survival_reward
    survival_mode: bool = False
    survival_reward: int = 1

    def deltas(self, result: str, n_steps: int, game_length: int | None = None,
               reference_length: float | None = None) -> list[int]:
        """回傳本局第 0..n_steps-1 個 AI 選擇的牌數增減量。

        survival_mode 時需提供 game_length（本局總手數）與 reference_length（過去輸棋的平均手數）。
        """
        if result == AI_WIN:
            return [self.win_reward] * n_steps
        if result == DRAW:
            return [self.draw_reward] * n_steps
        if result == AI_LOSS:
            if self.survival_mode:
                if reference_length is None or game_length is None or game_length == reference_length:
                    return [0] * n_steps
                if game_length > reference_length:
                    return [self.survival_reward] * n_steps
            d = [-self.loss_penalty] * n_steps
            if n_steps and self.final_loss_penalty is not None:
                d[-1] = -self.final_loss_penalty
            return d
        raise ValueError(f"未知的結果: {result}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RewardConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
