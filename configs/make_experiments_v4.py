"""產生 configs/experiments_v4.json：第四種模式「只靠規則」的正式實驗。

執行：python configs/make_experiments_v4.py

- AI：agents/rule_only_card_agent.py（3×3、2×4、4×2 小區塊原始內容；TD(λ)；八種對稱平均）。
- 訓練對手只有自我對弈（90%）與隨機（10%），先後手輪流，30 萬局，3 個亂數種子。
- 規則型、Minimax、Rapfi 只用來評估；攻防測驗只用來評分，AI 學習時看不到。
- 主要種子（2026）前 2000 局保存每一手的牌數變化，其餘只保存落子。
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [2026, 2027, 2028]
MAIN_SEED = 2026

AI = {"type": "rule_only_card", "config": {"shapes": [[3, 3], [2, 4], [4, 2]], "alpha": 0.1, "lam": 0.7,
                                            "beta": 10.0, "initial_weight": 10.0, "version": "rule-only-1.0"}}
SELF_RANDOM = {"type": "mixed", "name": "self+random", "pool": [
    {"type": "self", "weight": 0.9, "name": "self-play"},
    {"type": "random", "weight": 0.1, "name": "random"},
]}
RULE = {"type": "heuristic", "level": "basic", "deterministic": True, "name": "rule-deterministic"}
RULE_RANDOM = {"type": "heuristic", "level": "basic", "deterministic": False, "name": "rule-stochastic"}
PATTERN = {"type": "heuristic", "level": "pattern", "deterministic": True, "name": "pattern-deterministic"}
CHECKPOINTS = [
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


experiments = []
for s in SEEDS:
    experiments.append({
        "name": "rule_only_mixed" if s == MAIN_SEED else f"rule_only_mixed_seed{s}",
        "group": "rule_only_mixed",
        "description": "只靠規則：小區塊原始內容位子牌，自我對弈 90%＋隨機 10%，先後手輪流",
        "seed": s, "ai": AI, "opponent": SELF_RANDOM, "ai_color": "alternate", "train_games": 300000,
        "checkpoint_eval": CHECKPOINTS, "log_learning": s == MAIN_SEED,
        "log_learning_games": 2000 if s == MAIN_SEED else 0,
        "compress_logs": True, "compress_model": True,
    })

config = {
    "defaults": {"eval_every": 25000, "eval_games": 100, "segment": 25000, "rolling_window": 50,
                 "stable_threshold": 0.5, "final_eval_games": 200, "tactics": True, "final_eval": final_eval()},
    "experiments": experiments,
}

if __name__ == "__main__":
    with open(os.path.join(HERE, "experiments_v4.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"{len(experiments)} 組實驗")
