"""實驗執行器：依 JSON 設定檔進行「訓練前評估 → 訓練（定期評估）→ 訓練後評估」，並輸出所有紀錄。

輸出（name = 實驗名稱，既有檔案預設不覆蓋）：
    data/experiments/<name>/config.json      實際使用的設定（含時間、Python 版本）
    data/experiments/<name>/train.csv        每一局訓練的結果與累計/滾動勝率
    data/experiments/<name>/eval.csv         每個檢查點的凍結評估
    data/experiments/<name>/summary.json     首次勝利、穩定勝利、學習資料統計、最終評估
    data/matches/<name>_train.jsonl          每一局訓練棋譜（含選擇前後牌數）
    data/matches/<name>_eval.jsonl           每一局評估棋譜
    data/models/<name>.json                  訓練後的學習資料
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import shutil
import time

import numpy as np

from agents.base_agent import Agent
from agents.gradient_card_agent import GradientCardAgent, GradientCardConfig
from agents.heuristic_agent import HeuristicAgent
from agents.menace_agent import MenaceAgent, MenaceConfig
from agents.minimax_agent import MinimaxAgent
from agents.mixed_opponent import MixedOpponent
from agents.pattern_menace_agent import PatternMenaceAgent, PatternMenaceConfig
from agents.random_agent import RandomAgent
from agents.rule_only_card_agent import RuleOnlyCardAgent, RuleOnlyConfig
from agents.rapfi_agent import EngineUnavailableError, RapfiAgent
from core.board import NAME_COLOR
from core.symmetry import canonical
from reporting.logger import MatchLogger, write_csv

from .evaluation import evaluate
from .tactics import tactical_score
from .trainer import Tally, run_games

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def make_agent(spec: dict, seed: int | None = None, ai: Agent | None = None) -> Agent:
    """依設定建立代理。type="self" 需提供 ai（建立共用牌組的自我對弈分身）。"""
    t = spec["type"]
    name = spec.get("name")
    if t == "self":
        if ai is None or not hasattr(ai, "twin"):
            raise ValueError("自我對弈需要支援 twin() 的學習 AI")
        return ai.twin(name=name)
    if t == "mixed":
        pool = [(make_agent(e, None if seed is None else seed + 101 * (k + 1), ai), e.get("weight", 1.0))
                for k, e in enumerate(spec["pool"])]
        return MixedOpponent(pool, name=name, seed=seed)
    if t == "menace":
        if "load" in spec:
            return MenaceAgent.load(os.path.join(ROOT, spec["load"]), seed=seed, name=name)
        return MenaceAgent(MenaceConfig.from_dict(spec.get("config", {})), name=name, seed=seed)
    if t == "pattern_menace":
        if "load" in spec:
            return PatternMenaceAgent.load(os.path.join(ROOT, spec["load"]), seed=seed, name=name)
        return PatternMenaceAgent(PatternMenaceConfig.from_dict(spec.get("config", {})), name=name, seed=seed)
    if t == "card_pg":
        if "load" in spec:
            return GradientCardAgent.load(os.path.join(ROOT, spec["load"]), seed=seed, name=name)
        return GradientCardAgent(GradientCardConfig.from_dict(spec.get("config", {})), name=name, seed=seed)
    if t == "neural_card":
        from agents.neural_card_agent import NeuralCardAgent, NeuralCardConfig
        if "load" in spec:
            return NeuralCardAgent.load(os.path.join(ROOT, spec["load"]), seed=seed, name=name)
        return NeuralCardAgent(NeuralCardConfig.from_dict(spec.get("config", {})), name=name, seed=seed)
    if t == "rule_only_card":
        if "load" in spec:
            return RuleOnlyCardAgent.load(os.path.join(ROOT, spec["load"]), seed=seed, name=name)
        return RuleOnlyCardAgent(RuleOnlyConfig.from_dict(spec.get("config", {})), name=name, seed=seed)
    if t == "random":
        return RandomAgent(name=name, seed=seed)
    if t == "heuristic":
        return HeuristicAgent(spec.get("level", "pattern"), spec.get("deterministic", False),
                              spec.get("epsilon", 0.0), name=name, seed=seed)
    if t == "minimax":
        return MinimaxAgent(spec.get("depth", 2), spec.get("max_candidates", 8), name=name, seed=seed)
    if t == "rapfi":
        return RapfiAgent(spec.get("path"), spec.get("max_node", 20000), spec.get("max_depth"),
                          spec.get("timeout_turn_ms", 5000), name=name, seed=seed)
    raise ValueError(f"未知的代理類型: {t}")


def _paths(name: str, compress: bool = False, compress_model: bool = False, model_ext: str | None = None) -> dict:
    exp_dir = os.path.join(DATA, "experiments", name)
    ext = ".jsonl.gz" if compress else ".jsonl"
    return {
        "dir": exp_dir,
        "config": os.path.join(exp_dir, "config.json"),
        "train_csv": os.path.join(exp_dir, "train.csv"),
        "eval_csv": os.path.join(exp_dir, "eval.csv"),
        "summary": os.path.join(exp_dir, "summary.json"),
        "train_jsonl": os.path.join(DATA, "matches", f"{name}_train{ext}"),
        "eval_jsonl": os.path.join(DATA, "matches", f"{name}_eval{ext}"),
        "model": os.path.join(DATA, "models", f"{name}{model_ext}" if model_ext else
                              (f"{name}.json.gz" if compress_model else f"{name}.json")),
    }


def learning_stats(ai: Agent) -> dict:
    """學習資料的統計（報告用）。"""
    if getattr(ai, "kind", "") == "neural_card":
        return {"parameters": len(ai.memory), "games_trained": ai.games_trained,
                "channels": ai.config.channels, "blocks": ai.config.blocks}
    if isinstance(ai, MenaceAgent):
        visits = list(ai.memory.visits.values())
        stats = {**ai.memory.stats(), **ai.opponent_memory.stats()}
        stats["states_visited_once"] = sum(1 for v in visits if v == 1)
        stats["states_visited_2plus"] = sum(1 for v in visits if v >= 2)
        stats["states_visited_10plus"] = sum(1 for v in visits if v >= 10)
        stats["max_visits"] = max(visits) if visits else 0
        if not ai.config.use_symmetry:
            # 若改用對稱去重，同一批局面會合併成多少類
            decode = {"0": 0, "1": 1, "2": -1}
            classes = {canonical([decode[ch] for ch in k])[0] for k in ai.memory.states}
            stats["unique_states_after_symmetry"] = len(classes)
        return stats
    if isinstance(ai, RuleOnlyCardAgent):
        w = ai.memory.w[:-1]
        return {"features": int(len(w)), "features_learned": int(np.count_nonzero(w)),
                "windows": ai.t.n_windows, "window_classes": ai.t.n_classes,
                "max_weight": round(float(w.max()), 4), "min_weight": round(float(w.min()), 4),
                **ai.opponent_memory.stats()}
    if isinstance(ai, GradientCardAgent):
        cards = sorted(((ai.memory.card(k), k) for k in ai.memory.theta), reverse=True)
        return {**ai.memory.stats(), **ai.opponent_memory.stats(),
                "top_cards": [{"pattern": k, "card": round(v, 2)} for v, k in cards[:8]],
                "bottom_cards": [{"pattern": k, "card": round(v, 4)} for v, k in cards[-5:]]}
    if isinstance(ai, PatternMenaceAgent):
        cards = list(ai.memory.cards.values())
        init = ai.memory.initial_weight
        return {**ai.memory.stats(), **ai.opponent_memory.stats(),
                "patterns_above_initial": sum(1 for c in cards if c > init),
                "patterns_below_initial": sum(1 for c in cards if c < init),
                "max_card": max(cards) if cards else None}
    return {}


def run_experiment(cfg: dict, force: bool = False, quiet: bool = False) -> dict:
    name = cfg["name"]
    p = _paths(name, cfg.get("compress_logs", False), cfg.get("compress_model", False),
               ".pt" if cfg.get("ai", {}).get("type") == "neural_card" else None)
    if os.path.exists(p["summary"]) and not force:
        raise FileExistsError(f"實驗 {name} 已有結果；為保留原始數據不會覆蓋（請改名或加 --force）")
    if force:
        for key in ("train_jsonl", "eval_jsonl", "model"):
            if os.path.exists(p[key]):
                os.remove(p[key])
        if os.path.isdir(p["dir"]):
            shutil.rmtree(p["dir"])
    os.makedirs(p["dir"], exist_ok=True)

    seed = cfg.get("seed", 2026)
    ai_color_name = cfg.get("ai_color", "black")
    ai_color = "alternate" if ai_color_name == "alternate" else NAME_COLOR[ai_color_name]
    eval_color = NAME_COLOR["black"] if ai_color == "alternate" else ai_color
    n_train = cfg.get("train_games", 5000)
    eval_every = cfg.get("eval_every", 500)
    eval_games = cfg.get("eval_games", 100)
    final_eval_games = cfg.get("final_eval_games", 200)
    log_learning = cfg.get("log_learning", True)

    meta = {**cfg, "_started": _dt.datetime.now().isoformat(timespec="seconds"),
            "_python": platform.python_version(), "_platform": platform.platform()}
    with open(p["config"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    ai = make_agent(cfg["ai"], seed=seed)
    train_opp = make_agent(cfg["opponent"], seed=seed + 1, ai=ai)
    # 檢查點評估的對手：預設與訓練對手相同；訓練對手含自我對弈時必須另外指定固定對手
    cp_specs = cfg.get("checkpoint_eval") or [{"spec": cfg["opponent"], "ai_color": None, "label": "checkpoint"}]
    cp_opps = []
    for e in cp_specs:
        opp = train_opp if e["spec"] is cfg["opponent"] else make_agent(e["spec"], seed=seed + 3)
        color = eval_color if e.get("ai_color") is None else NAME_COLOR[e["ai_color"]]
        cp_opps.append((e.get("label", opp.name), opp, color))
    use_tactics = cfg.get("tactics", False) and hasattr(ai, "move_probabilities")
    batched = hasattr(ai, "train_games")
    if batched:
        import torch
        torch.set_num_threads(cfg.get("torch_threads", 1))
    t0 = time.time()

    def log(msg: str) -> None:
        if not quiet:
            print(f"[{name}] {msg}", flush=True)

    eval_rows: list[dict] = []
    eval_index = [0]

    def checkpoint(after: int, eval_log: MatchLogger) -> None:
        tac = tactical_score(ai) if use_tactics else None
        for label, opp, color in cp_opps:
            res = evaluate(ai, opp, eval_games, ai_color=color, base_seed=seed + 7, experiment=name,
                           logger=eval_log, start_index=eval_index[0] + 1, label=label)
            eval_index[0] += eval_games
            row = {"after_train_games": after, "label": label, "opponent": opp.name,
                   "ai_color": "black" if color == NAME_COLOR["black"] else "white", "games": res["games"],
                   "wins": res["wins"], "losses": res["losses"], "draws": res["draws"], "win_rate": res["win_rate"],
                   "avg_moves": res["avg_moves"]}
            if tac:
                row.update({f"tactics_{k}": tac[k]["prob"] for k in ("win", "block", "three")})
            eval_rows.append(row)
            log(f"檢查點 {after:>6} 局 [{label}]：評估勝率 {res['win_rate']:.3f}（{res['wins']}/{res['games']}）"
                + (f"；攻防測驗 連五 {tac['win']['prob']:.2f} 擋四 {tac['block']['prob']:.2f} "
                   f"擋活三 {tac['three']['prob']:.2f}" if tac else ""))

    tally = Tally(window=cfg.get("rolling_window", 50), stable_threshold=cfg.get("stable_threshold", 0.5))
    train_rows: list[dict] = []
    with MatchLogger(p["train_jsonl"]) as train_log, MatchLogger(p["eval_jsonl"]) as eval_log:
        checkpoint(0, eval_log)
        done = 0
        log_until = cfg.get("log_learning_games", n_train if log_learning else 0)
        while done < n_train:
            chunk = min(eval_every - done % eval_every, n_train - done)
            detailed = done < log_until
            if detailed:
                chunk = min(chunk, log_until - done)
            if hasattr(ai, "record_after"):
                ai.record_after = detailed
            if batched:   # 批次自我對弈（第五種模式）：同時下多局、每批學習一次
                rows = ai.train_games(chunk, base_seed=seed, start_index=done + 1,
                                      random_frac=cfg.get("random_frac", 0.1), logger=train_log,
                                      experiment=name, include_learning=detailed, tally=tally)
            else:
                rows = run_games(ai, train_opp, chunk, ai_color=ai_color, base_seed=seed, phase="train",
                                 experiment=name, logger=train_log, start_index=done + 1, tally=tally,
                                 include_learning=detailed)
            train_rows.extend(rows)
            done += chunk
            if done % eval_every == 0 or done == n_train:
                log(f"訓練 {done}/{n_train}：累計勝率 {tally.wins / tally.n:.3f}，首次勝利 {tally.first_win}")
                checkpoint(done, eval_log)

        final = []
        for spec in cfg.get("final_eval", []):
            label = spec.get("label") or spec.get("name") or spec["type"]
            try:
                opp = make_agent(spec, seed=seed + 2)
            except EngineUnavailableError as e:
                final.append({"label": label, "skipped": str(e)})
                log(f"略過最終評估 {label}：{e}")
                continue
            color = eval_color if spec.get("ai_color") is None else NAME_COLOR[spec["ai_color"]]
            res = evaluate(ai, opp, spec.get("games", final_eval_games), ai_color=color, base_seed=seed + 11,
                           experiment=name, logger=eval_log, start_index=eval_index[0] + 1, label=label)
            eval_index[0] += res["games"]
            final.append(res)
            log(f"最終評估 vs {label}：{res['wins']} 勝 {res['losses']} 敗 {res['draws']} 和")
            if hasattr(opp, "close"):
                opp.close()

    if hasattr(train_opp, "close"):
        train_opp.close()
    write_csv(p["train_csv"], train_rows, overwrite=True)
    write_csv(p["eval_csv"], eval_rows, overwrite=True)
    if hasattr(ai, "save"):
        ai.save(p["model"])

    # 分段勝率（每 segment 局一段）
    seg = cfg.get("segment", 500)
    segments = []
    for i in range(0, len(train_rows), seg):
        part = train_rows[i:i + seg]
        segments.append({"games": f"{i + 1}-{i + len(part)}",
                         "win_rate": round(sum(r["result"] == "ai_win" for r in part) / len(part), 4),
                         "loss_rate": round(sum(r["result"] == "ai_loss" for r in part) / len(part), 4)})
    summary = {
        "name": name,
        "group": cfg.get("group", name),
        "description": cfg.get("description", ""),
        "ai": ai.describe(),
        "opponent": train_opp.describe(),
        "ai_color": cfg.get("ai_color", "black"),
        "seed": seed,
        "training": tally.summary(),
        "training_segments": segments,
        "checkpoints": eval_rows,
        "final_eval": final,
        "learning_stats": learning_stats(ai),
        **({"tactics": tactical_score(ai)} if use_tactics else {}),
        "elapsed_sec": round(time.time() - t0, 1),
        "files": {k: os.path.relpath(v, ROOT) for k, v in p.items() if k != "dir"},
    }
    with open(p["summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"完成，用時 {summary['elapsed_sec']} 秒")
    return summary


def load_config(path: str) -> list[dict]:
    """設定檔可以是單一實驗或 {"experiments": [...]}，後者可用 "defaults" 共用欄位。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "experiments" in data:
        defaults = data.get("defaults", {})
        return [{**defaults, **e} for e in data["experiments"]]
    return [data]
