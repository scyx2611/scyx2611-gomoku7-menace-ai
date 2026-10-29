"""產生 configs/experiments.json（實驗設計集中在這裡，方便檢查與修改）。

執行：python configs/make_experiments.py

設計原則
- 所有實驗使用同一個訓練對手：確定性規則型對手（能連五就連、對手要連五就擋，
  否則下最靠近中央的空位），同一局面一定下同一手。
- 每組訓練 10000 局；每 500 局用凍結權重評估 100 局。
- 學習突破的時間點受亂數影響很大，所以每種設定都跑多個種子，報告列出全部結果。
- 主要實驗（種子 2026）保存含牌數變化的完整棋譜；其他實驗只保存落子（gzip 壓縮）。
"""

import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REWARD = {"win_reward": 3, "loss_penalty": 1, "draw_reward": 1, "initial_weight": 10, "min_weight": 1}
BASE = {"type": "menace", "config": {"reward": REWARD, "use_symmetry": False, "opponent_model": "off",
                                     "version": "menace-1.0"}}
IMP = {"type": "pattern_menace", "config": {"reward": REWARD, "combine": "product", "radius": 4,
                                            "use_symmetry": True, "opponent_model": "off",
                                            "version": "pattern-menace-1.0"}}
RULE = {"type": "heuristic", "level": "basic", "deterministic": True, "name": "rule-deterministic"}
RANDOM = {"type": "random", "name": "random"}
MAIN_SEED = 2026
SEEDS5 = [2026, 2027, 2028, 2029, 2030]
SEEDS3 = [2026, 2027, 2028]


def variant(base, **kw):
    a = copy.deepcopy(base)
    for k, v in kw.items():
        if k == "reward":
            a["config"]["reward"].update(v)
        else:
            a["config"][k] = v
    return a


def group(name, desc, ai, seeds, color="black", opp=RULE, full_log_main=False):
    out = []
    for s in seeds:
        main = full_log_main and s == MAIN_SEED
        out.append({"name": name if s == MAIN_SEED else f"{name}_seed{s}", "group": name, "description": desc,
                    "seed": s, "ai": ai, "ai_color": color, "opponent": opp,
                    "log_learning": main, "compress_logs": not main})
    return out


experiments = (
    group("baseline_first", "修正前：整盤局面位子牌（MENACE 原型），先手", BASE, SEEDS5, full_log_main=True)
    + group("improved_first", "修正後：棋形位子牌（四方向牌數相乘），先手", IMP, SEEDS5, full_log_main=True)
    + group("baseline_second", "修正前，後手", BASE, SEEDS3, "white", full_log_main=True)
    + group("improved_second", "修正後，後手", IMP, SEEDS3, "white", full_log_main=True)
    + group("baseline_symmetry_first", "修正前 + 八種對稱去重", variant(BASE, use_symmetry=True), SEEDS3)
    + group("improved_nosym_first", "修正後但關閉對稱合併（四方向不共用、正反不合併）",
            variant(IMP, use_symmetry=False), SEEDS3)
    + group("baseline_oppavoid_first", "修正前 + 對手記憶避讓（避開對手曾直接五連的局面）",
            variant(BASE, opponent_model="avoid"), SEEDS3)
    + group("improved_oppmem_first", "修正後 + 對手記憶搶點（提高對手偏好棋形位置的牌數）",
            variant(IMP, opponent_model="preempt"), SEEDS3)
    + group("improved_sum_first", "修正後，但四方向牌數改為相加", variant(IMP, combine="sum"), SEEDS3)
    + group("improved_noloss_first", "修正後，只獎勵不懲罰（敗 0）", variant(IMP, reward={"loss_penalty": 0}), SEEDS3)
    + group("improved_nowin_first", "修正後，只懲罰不獎勵（勝 0）", variant(IMP, reward={"win_reward": 0}), SEEDS3)
    + group("improved_heavyloss_first", "修正後，加重懲罰（敗 -3）", variant(IMP, reward={"loss_penalty": 3}), SEEDS3)
    + group("improved_survival_first", "修正後 + 存活獎勵（輸棋但比平均撐得久 +1）",
            variant(IMP, reward={"survival_mode": True}), SEEDS3)
    + group("baseline_vs_random_first", "修正前，對隨機對手", BASE, [MAIN_SEED], opp=RANDOM)
    + group("improved_vs_random_first", "修正後，對隨機對手", IMP, [MAIN_SEED], opp=RANDOM)
)

config = {
    "defaults": {
        "train_games": 10000, "eval_every": 500, "eval_games": 100, "segment": 500,
        "rolling_window": 50, "stable_threshold": 0.5, "final_eval_games": 200,
        "final_eval": [
            {"type": "heuristic", "level": "basic", "deterministic": True, "label": "規則型（確定性，訓練對手）"},
            {"type": "heuristic", "level": "basic", "deterministic": False, "label": "規則型（隨機版）"},
            {"type": "heuristic", "level": "pattern", "deterministic": True, "label": "棋形計分規則型"},
            {"type": "random", "label": "隨機"},
            {"type": "minimax", "depth": 2, "label": "Minimax 深度 2"},
            {"type": "rapfi", "max_node": 20000, "label": "Rapfi（外部引擎）", "games": 100},
        ],
    },
    "experiments": experiments,
}

if __name__ == "__main__":
    with open(os.path.join(HERE, "experiments.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"{len(experiments)} 組實驗")
