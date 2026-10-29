"""評估：以「凍結」的 AI 副本對弈，保證不改動受測的學習資料。"""

from __future__ import annotations

import hashlib
import json

from agents.base_agent import Agent
from core.board import BLACK

from .trainer import Tally, is_learning_agent, run_games


def fingerprint(agent: Agent) -> str:
    """學習資料（牌組＋對手記憶）的雜湊值，用來檢查評估前後是否一致。"""
    data = {"memory": agent.memory.to_dict(), "opp": agent.opponent_memory.to_dict()}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def frozen(ai: Agent) -> Agent:
    return ai.clone(learning=False) if is_learning_agent(ai) else ai


def evaluate(ai: Agent, opponent: Agent, n_games: int, *, ai_color: int = BLACK, base_seed: int = 0,
             experiment: str = "eval", logger=None, start_index: int = 1, label: str | None = None) -> dict:
    """回傳評估摘要（勝/敗/和、勝率、平均手數）。原始 ai 物件的學習資料保證不變。"""
    before = fingerprint(ai) if is_learning_agent(ai) else None
    player = frozen(ai)
    tally = Tally()
    rows = run_games(player, opponent, n_games, ai_color=ai_color, base_seed=base_seed, phase="eval",
                     experiment=experiment, logger=logger, start_index=start_index, tally=tally,
                     include_learning=False)
    if before is not None and fingerprint(ai) != before:
        raise RuntimeError("評估過程改動了 AI 的學習資料")
    s = tally.summary()
    return {
        "label": label or opponent.name,
        "opponent": opponent.describe(),
        "ai_color": "black" if ai_color == BLACK else "white",
        "games": s["games"], "wins": s["wins"], "losses": s["losses"], "draws": s["draws"],
        "win_rate": s["win_rate"],
        "avg_moves": round(sum(r["move_count"] for r in rows) / len(rows), 2) if rows else None,
    }
