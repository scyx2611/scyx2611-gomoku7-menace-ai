"""產生 configs/experiments_v5.json：第五種模式「位子牌＋神經網路（只靠規則）」的正式實驗。

執行：python configs/make_experiments_v5.py（需要 PyTorch）

- AI：agents/neural_card_agent.py（16 通道 × 3 層殘差卷積網路估「下完後盤面」勝算；β = 10；TD(λ = 0.7)）。
- 訓練：批次自我對弈（每批 64 局），每局 10% 機率改與隨機對手下；30 萬局，3 個亂數種子。
- 規則型、Minimax、Rapfi 只用來評估；攻防測驗只用來評分。評估、檢查點與第四種模式完全相同，方便比較。
- 主要種子（2026）前 2000 局保存每一手的估值、牌數與學習後牌數。
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
import make_experiments_v4 as v4  # noqa: E402

SEEDS = [2026, 2027, 2028]
AI = {"type": "neural_card", "config": {"channels": 16, "blocks": 3, "beta": 10.0, "lam": 0.7, "lr": 1e-3,
                                         "batch_games": 64, "initial_weight": 10.0, "version": "neural-card-1.0"}}

experiments = [{
    "name": "neural_mixed" if s == 2026 else f"neural_mixed_seed{s}",
    "group": "neural_mixed",
    "description": "神經網路（只靠規則）：估下完後盤面，批次自我對弈＋10% 隨機",
    "seed": s, "ai": AI, "opponent": v4.SELF_RANDOM, "ai_color": "alternate", "random_frac": 0.1,
    "train_games": 300000, "checkpoint_eval": v4.CHECKPOINTS, "log_learning": s == 2026,
    "log_learning_games": 2000 if s == 2026 else 0, "compress_logs": True, "torch_threads": 1,
} for s in SEEDS]

config = {"defaults": {**json.load(open(os.path.join(HERE, "experiments_v4.json"), encoding="utf-8"))["defaults"]},
          "experiments": experiments}

if __name__ == "__main__":
    with open(os.path.join(HERE, "experiments_v5.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"{len(experiments)} 組實驗")
