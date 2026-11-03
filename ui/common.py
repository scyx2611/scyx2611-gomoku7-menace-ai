"""人機對弈共用：可選擇的 AI 清單、對局紀錄保存。"""

from __future__ import annotations

import datetime as _dt
import gzip
import json
import os

from agents.base_agent import Agent
from agents.gradient_card_agent import GradientCardAgent
from agents.heuristic_agent import HeuristicAgent
from agents.menace_agent import MenaceAgent
from agents.minimax_agent import MinimaxAgent
from agents.pattern_menace_agent import PatternMenaceAgent
from agents.random_agent import RandomAgent
from agents.rule_only_card_agent import RuleOnlyCardAgent
from agents.rapfi_agent import RapfiAgent, rapfi_available
from core.board import COLOR_NAME
from core.game import Game

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARD_MODEL = os.path.join(ROOT, "data", "models", "card_v3_mixed.json")
IMPROVED_MODEL = os.path.join(ROOT, "data", "models", "improved_first.json")
RULE_ONLY_MODEL = os.path.join(ROOT, "data", "models", "rule_only_mixed.json.gz")
BASELINE_MODEL = os.path.join(ROOT, "data", "models", "baseline_first.json")
HUMAN_LOG = os.path.join(ROOT, "data", "matches", "human_games.jsonl")


def _load_or_new(cls, path: str):
    if os.path.exists(path):
        return cls.load(path, learning=False), True
    return cls(learning=False), False


# (代號, 顯示名稱, 建立函式) —— 建立函式回傳 (agent, 說明)
def ai_choices(model: str | None = None) -> list[tuple[str, str, callable]]:
    def learned():
        path = model or CARD_MODEL
        if not os.path.exists(path):
            return GradientCardAgent(learning=False), "尚無訓練資料（未訓練）"
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        cls = {"pattern-menace-weights": PatternMenaceAgent, "card-pg-weights": GradientCardAgent,
               "rule-only-card-weights": RuleOnlyCardAgent}.get(data.get("format"), MenaceAgent)
        return cls.from_dict(data, learning=False), f"已載入 {os.path.basename(path)}"

    def rule_only():
        if os.path.exists(RULE_ONLY_MODEL):
            return RuleOnlyCardAgent.load(RULE_ONLY_MODEL, learning=False), "只靠規則，自我對弈學習"
        return RuleOnlyCardAgent(learning=False), "尚無訓練資料（未訓練）"

    def improved():
        a, ok = _load_or_new(PatternMenaceAgent, IMPROVED_MODEL)
        return a, "已載入第一次修正模型" if ok else "尚無訓練資料（未訓練）"

    def baseline():
        a, ok = _load_or_new(MenaceAgent, BASELINE_MODEL)
        return a, "已載入修正前模型" if ok else "尚無訓練資料（未訓練）"

    choices = [
        ("learned", "位子牌 AI（第二次修正）", learned),
        ("ruleonly", "位子牌 AI（只靠規則）", rule_only),
        ("improved", "位子牌 AI（第一次修正）", improved),
        ("baseline", "位子牌 AI（修正前）", baseline),
        ("rule", "規則型對手", lambda: (HeuristicAgent("basic", deterministic=True), "只會連五與擋五")),
        ("pattern", "棋形計分規則型", lambda: (HeuristicAgent("pattern", deterministic=True), "依棋形計分")),
        ("minimax", "Minimax（搜尋）", lambda: (MinimaxAgent(depth=2), "Alpha-Beta 深度 2，非學習")),
        ("random", "隨機", lambda: (RandomAgent(), "隨機落子")),
    ]
    if rapfi_available():
        choices.append(("rapfi", "Rapfi（外部引擎）", lambda: (RapfiAgent(max_node=200000), "外部引擎，非學習")))
    return choices


def save_human_game(game: Game, human_color: int, ai: Agent) -> None:
    os.makedirs(os.path.dirname(HUMAN_LOG), exist_ok=True)
    rec = {
        "game_id": "human-" + _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
        "experiment": "human_play",
        "phase": "play",
        "ai": ai.describe(),
        "human_color": COLOR_NAME[human_color],
        "ai_color": COLOR_NAME[-human_color],
        "moves": [{**m.to_dict(), "by": "human" if m.color == human_color else "ai"} for m in game.history],
        "move_count": len(game.history),
        "winner": COLOR_NAME.get(game.winner) if game.winner else None,
        "result": "draw" if game.winner is None else ("ai_win" if game.winner == -human_color else "ai_loss"),
    }
    with open(HUMAN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
