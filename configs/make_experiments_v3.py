"""產生 configs/experiments_v3.json：第二次修正（第三版 AI）的正式實驗。

執行：python configs/make_experiments_v3.py

第三版 AI：棋形計數牌（features="window"）＋按比例增減牌數（策略梯度）。
- card_v3_rule_first / _second：與前兩版完全相同的條件（只對確定性規則型，10000 局），直接比較由敗至贏。
- card_v3_mixed：混合對手（自我對弈 50%、隨機版規則型 20%、帶隨機的棋形計分規則型 20%、隨機 10%），
  先後手輪流，20000 局。對弈介面預設使用這個模型。
- card_v3_selfplay：純自我對弈。
- 拆解實驗（找出是哪個修正有效）：
    card_v3_line_mixed      按比例增減，但用第二版的完整排列棋形
    window_additive_mixed   視窗計數棋形，但用第二版的整數加減牌（勝 +1、敗 −1）
- 攻防測驗（training/tactics.py）在每個檢查點與訓練結束時計算。
- 前兩版的實驗（configs/experiments.json）不受影響。
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [2026, 2027, 2028]
MAIN_SEED = 2026

CARD = {"type": "card_pg", "config": {"initial_weight": 10.0, "alpha": 0.1, "gamma": 0.9, "win_return": 1.0,
                                       "loss_return": -1.0, "draw_return": 0.0, "features": "window",
                                       "version": "card-pg-1.0"}}
CARD_LINE = {"type": "card_pg", "config": {**CARD["config"], "features": "line", "version": "card-pg-1.0-line"}}
ADDITIVE_WINDOW = {"type": "pattern_menace", "config": {
    "reward": {"win_reward": 1, "loss_penalty": 1, "draw_reward": 0, "initial_weight": 10, "min_weight": 1},
    "combine": "product", "features": "window", "use_symmetry": True, "version": "pattern-menace-1.0-window"}}

RULE = {"type": "heuristic", "level": "basic", "deterministic": True, "name": "rule-deterministic"}
RULE_RANDOM = {"type": "heuristic", "level": "basic", "deterministic": False, "name": "rule-stochastic"}
PATTERN = {"type": "heuristic", "level": "pattern", "deterministic": True, "name": "pattern-deterministic"}
MIXED = {"type": "mixed", "name": "mixed", "pool": [
    {"type": "self", "weight": 0.5, "name": "self-play"},
    {"type": "heuristic", "level": "basic", "deterministic": False, "weight": 0.2, "name": "rule-stochastic"},
    {"type": "heuristic", "level": "pattern", "deterministic": False, "epsilon": 0.2, "weight": 0.2,
     "name": "pattern-eps0.2"},
    {"type": "random", "weight": 0.1, "name": "random"},
]}

CP_GENERAL = [
    {"spec": RULE_RANDOM, "ai_color": "black", "label": "規則型隨機版・AI 先手"},
    {"spec": RULE_RANDOM, "ai_color": "white", "label": "規則型隨機版・AI 後手"},
    {"spec": PATTERN, "ai_color": "black", "label": "棋形計分規則型・AI 先手"},
]


def final_eval():
    out = []
    for spec, label in [(RULE, "規則型（確定性）"), (RULE_RANDOM, "規則型（隨機版）"), (PATTERN, "棋形計分規則型"),
                        ({"type": "heuristic", "level": "pattern", "deterministic": False}, "棋形計分規則型（隨機版）"),
                        ({"type": "random"}, "隨機"), ({"type": "minimax", "depth": 2}, "Minimax 深度 2"),
                        ({"type": "rapfi", "max_node": 20000, "games": 100}, "Rapfi（外部引擎）")]:
        for color, cname in (("black", "AI 先手"), ("white", "AI 後手")):
            out.append({**spec, "ai_color": color, "label": f"{label}・{cname}"})
    return out


def group(name, desc, ai, opp, color, games, seeds=SEEDS, checkpoint=None, full_log=False):
    out = []
    for s in seeds:
        main = full_log and s == MAIN_SEED
        e = {"name": name if s == MAIN_SEED else f"{name}_seed{s}", "group": name, "description": desc, "seed": s,
             "ai": ai, "opponent": opp, "ai_color": color, "train_games": games,
             "log_learning": main, "compress_logs": not main}
        if checkpoint:
            e["checkpoint_eval"] = checkpoint
        out.append(e)
    return out


experiments = (
    group("card_v3_rule_first", "第二次修正：只對確定性規則型、AI 先手（與前兩版相同條件）", CARD, RULE, "black", 10000,
          full_log=True)
    + group("card_v3_rule_second", "第二次修正：只對確定性規則型、AI 後手", CARD, RULE, "white", 10000)
    + group("card_v3_mixed", "第二次修正：混合對手＋自我對弈、先後手輪流", CARD, MIXED, "alternate", 20000,
            checkpoint=CP_GENERAL, full_log=True)
    + group("card_v3_selfplay", "第二次修正：純自我對弈、先後手輪流", CARD, {"type": "self", "name": "self-play"},
            "alternate", 20000, checkpoint=CP_GENERAL)
    + group("card_v3_line_mixed", "拆解：按比例增減，但棋形用第二版的完整排列", CARD_LINE, MIXED, "alternate", 20000,
            checkpoint=CP_GENERAL)
    + group("window_additive_mixed", "拆解：視窗計數棋形，但用第二版的整數加減牌（勝 +1、敗 -1）", ADDITIVE_WINDOW,
            MIXED, "alternate", 20000, checkpoint=CP_GENERAL)
)

config = {
    "defaults": {"eval_every": 1000, "eval_games": 100, "segment": 1000, "rolling_window": 50,
                 "stable_threshold": 0.5, "final_eval_games": 200, "tactics": True, "final_eval": final_eval()},
    "experiments": experiments,
}

if __name__ == "__main__":
    with open(os.path.join(HERE, "experiments_v3.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"{len(experiments)} 組實驗")
