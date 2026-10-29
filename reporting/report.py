"""由 data/experiments/*/summary.json 與 train.csv 產生圖表與 data/reports/experiment_report.md。

報告中的每個數字都直接讀自實驗輸出檔，不手動填寫。
"""

from __future__ import annotations

import csv
import glob
import json
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from core.game import replay  # noqa: E402
from reporting.logger import read_matches  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "data", "experiments")
OUT = os.path.join(ROOT, "data", "reports")

# 圖表配色（依角色固定，不隨排序改變）
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
C_IMPROVED = "#2a78d6"   # 修正後
C_BASELINE = "#eb6834"   # 修正前
C_THIRD = "#1baf7a"
ROLL = 200

GROUP_LABEL = {
    "baseline_first": "修正前・先手",
    "improved_first": "修正後・先手",
    "baseline_second": "修正前・後手",
    "improved_second": "修正後・後手",
    "baseline_symmetry_first": "修正前＋對稱去重",
    "improved_nosym_first": "修正後・關閉對稱合併",
    "baseline_oppavoid_first": "修正前＋對手記憶避讓",
    "improved_oppmem_first": "修正後＋對手記憶搶點",
    "improved_sum_first": "修正後・牌數相加",
    "improved_noloss_first": "修正後・只獎勵（敗 0）",
    "improved_nowin_first": "修正後・只懲罰（勝 0）",
    "improved_heavyloss_first": "修正後・加重懲罰（敗 -3）",
    "improved_survival_first": "修正後＋存活獎勵",
    "baseline_vs_random_first": "修正前・對隨機",
    "improved_vs_random_first": "修正後・對隨機",
}


def _setup_font() -> None:
    for name in ["Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans TC", "PingFang TC",
                 "WenQuanYi Zen Hei", "SimHei", "Arial Unicode MS"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except Exception:
            continue
        plt.rcParams["font.family"] = name
        break
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
    })


# ---- 讀取資料 ----------------------------------------------------------------
def load_summaries() -> dict[str, dict]:
    out = {}
    for path in sorted(glob.glob(os.path.join(EXP, "*", "summary.json"))):
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        out[s["name"]] = s
    return out


def load_train(name: str) -> list[dict]:
    with open(os.path.join(EXP, name, "train.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rolling(values: list[float], k: int) -> list[float]:
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= k:
            acc -= values[i - k]
        out.append(acc / min(i + 1, k))
    return out


def by_group(summaries: dict[str, dict]) -> dict[str, list[dict]]:
    g = defaultdict(list)
    for s in summaries.values():
        g[s.get("group", s["name"])].append(s)
    for v in g.values():
        v.sort(key=lambda s: s["seed"])
    return g


def last_rate(rows: list[dict], n: int = 2000) -> float:
    part = rows[-n:]
    return sum(r["result"] == "ai_win" for r in part) / len(part)


def final_eval(s: dict, label_prefix: str) -> dict | None:
    for e in s.get("final_eval", []):
        if e.get("label", "").startswith(label_prefix):
            return e
    return None


def fmt_rate(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def fmt(x) -> str:
    return "未出現" if x is None else str(x)


# ---- 圖表 ------------------------------------------------------------------
def _curve(ax, rows, color, label, lw=2.0, alpha=1.0, direct=True):
    wins = [1.0 if r["result"] == "ai_win" else 0.0 for r in rows]
    ys = [y * 100 for y in rolling(wins, ROLL)]
    xs = list(range(1, len(ys) + 1))
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, label=label)
    if direct and label:
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", color=INK2,
                    va="center", fontsize=9)


def chart_main(groups, path):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for gname, color in (("baseline_first", C_BASELINE), ("improved_first", C_IMPROVED)):
        for s in groups.get(gname, []):
            if s["name"] == gname:
                _curve(ax, load_train(s["name"]), color, GROUP_LABEL[gname])
                cp = s["checkpoints"]
                ax.plot([c["after_train_games"] for c in cp], [c["win_rate"] * 100 for c in cp], "o",
                        ms=5, color=color, mec=SURFACE, mew=1.5)
    ax.set_xlabel("訓練局數")
    ax.set_ylabel(f"勝率（%，最近 {ROLL} 局）")
    ax.set_ylim(-3, 103)
    ax.set_title("主要實驗（種子 2026）：訓練勝率；圓點為凍結權重評估（100 局）", color=INK, loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _label_ends(ax, ends: list[tuple[float, float, str]], min_gap: float = 5.0) -> None:
    """在線條末端標註名稱；末端太接近時往上下推開，避免文字重疊。"""
    ends = sorted(ends, key=lambda e: e[1])
    placed: list[float] = []
    for x, y, text in ends:
        ty = y if not placed else max(y, placed[-1] + min_gap)
        placed.append(ty)
        ax.annotate(text, (x, y), xytext=(x, ty), textcoords="data", color=INK2, va="center", fontsize=9,
                    xycoords="data", annotation_clip=False,
                    horizontalalignment="left").set_x(x)


def chart_seeds(groups, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, gname, color in ((axes[0], "baseline_first", C_BASELINE), (axes[1], "improved_first", C_IMPROVED)):
        ends = []
        for s in groups.get(gname, []):
            rows = load_train(s["name"])
            _curve(ax, rows, color, None, lw=1.4, alpha=0.85, direct=False)
            wins = [1.0 if r["result"] == "ai_win" else 0.0 for r in rows]
            ends.append((len(rows) * 1.02, rolling(wins, ROLL)[-1] * 100, f"種子 {s['seed']}"))
        if gname.startswith("baseline"):
            ax.annotate(f"{len(ends)} 個種子都貼近 0%", (len(rows) * 0.5, 6), color=INK2, ha="center", fontsize=9)
        else:
            _label_ends(ax, ends)
        ax.set_title(f"{GROUP_LABEL[gname]}：各亂數種子", color=INK, loc="left")
        ax.set_xlabel("訓練局數")
        ax.set_ylim(-3, 103)
    axes[0].set_ylabel(f"勝率（%，最近 {ROLL} 局）")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_groups(groups, names, path, title, colors=None):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    palette = colors or [C_IMPROVED, C_BASELINE, C_THIRD]
    for gname, color in zip(names, palette):
        for s in groups.get(gname, []):
            if s["seed"] == 2026:
                _curve(ax, load_train(s["name"]), color, GROUP_LABEL[gname])
    ax.set_xlabel("訓練局數")
    ax.set_ylabel(f"勝率（%，最近 {ROLL} 局）")
    ax.set_ylim(-3, 103)
    ax.set_title(title, color=INK, loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_dotplot(groups, names, path, title):
    """每個設定一列，點 = 各種子最後 2000 局訓練勝率，直線 = 平均。"""
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 1.4))
    for i, gname in enumerate(reversed(names)):
        runs = groups.get(gname, [])
        if not runs:
            continue
        vals = [last_rate(load_train(s["name"])) * 100 for s in runs]
        color = C_BASELINE if gname.startswith("baseline") else C_IMPROVED
        ax.scatter(vals, [i] * len(vals), s=55, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        m = statistics.mean(vals)
        ax.plot([m, m], [i - 0.28, i + 0.28], color=INK, lw=2, zorder=4)
        ax.annotate(f"平均 {m:.1f}%", (m, i + 0.3), color=INK2, fontsize=8, ha="center", va="bottom")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([GROUP_LABEL.get(n, n) for n in reversed(names)])
    ax.set_xlim(-3, 103)
    ax.set_ylim(-0.6, len(names) - 0.1)
    ax.set_xlabel("最後 2000 局訓練勝率（%）；圓點 = 各亂數種子，黑線 = 平均")
    ax.set_title(title, color=INK, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_length(groups, names, path):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for gname, color in zip(names, [C_IMPROVED, C_THIRD, C_BASELINE]):
        for s in groups.get(gname, []):
            if s["seed"] == 2026:
                rows = load_train(s["name"])
                mc = [int(r["move_count"]) for r in rows]
                ys = rolling(mc, ROLL)
                ax.plot(range(1, len(ys) + 1), ys, color=color, lw=2, label=GROUP_LABEL[gname])
                ax.annotate(GROUP_LABEL[gname], (len(ys), ys[-1]), xytext=(6, 0), textcoords="offset points",
                            color=INK2, va="center", fontsize=9)
    ax.set_xlabel("訓練局數")
    ax.set_ylabel(f"平均每局手數（最近 {ROLL} 局）")
    ax.set_title("每局手數：存活獎勵讓對局變長，但不代表會贏", color=INK, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---- 表格 ------------------------------------------------------------------
def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def seed_table(groups, names) -> str:
    rows = []
    for gname in names:
        for s in groups.get(gname, []):
            tr = s["training"]
            tr_rows = load_train(s["name"])
            fe = final_eval(s, "規則型（確定性")
            rows.append([GROUP_LABEL.get(gname, gname), s["seed"], tr["wins"], fmt(tr["first_win_game"]),
                         fmt(tr["stable_win_game"]), fmt_rate(sum(r["result"] == "ai_win" for r in tr_rows[:500]) / 500),
                         fmt_rate(last_rate(tr_rows)), fmt_rate(fe["win_rate"]) if fe and "win_rate" in fe else "—"])
    return table(["設定", "種子", "訓練勝場", "首次勝利（第幾局）", "穩定勝利（第幾局）", "前 500 局勝率",
                  "最後 2000 局勝率", "訓練後評估勝率（200 局）"], rows)


def group_mean_table(groups, names) -> str:
    rows = []
    for gname in names:
        runs = groups.get(gname, [])
        if not runs:
            continue
        lasts = [last_rate(load_train(s["name"])) for s in runs]
        firsts = [s["training"]["first_win_game"] for s in runs]
        stables = [s["training"]["stable_win_game"] for s in runs]
        reached = sum(1 for x in stables if x is not None)
        rows.append([GROUP_LABEL.get(gname, gname), len(runs), fmt_rate(statistics.mean(lasts)),
                     f"{fmt_rate(min(lasts))} ~ {fmt_rate(max(lasts))}",
                     f"{reached}/{len(runs)}",
                     ", ".join(fmt(x) for x in firsts)])
    return table(["設定", "種子數", "最後 2000 局平均勝率", "範圍", "達到穩定勝利的種子", "各種子首次勝利局號"], rows)


def final_eval_table(s: dict) -> str:
    rows = []
    for e in s.get("final_eval", []):
        if "skipped" in e:
            rows.append([e["label"], "略過", "", "", "", e["skipped"]])
        else:
            rows.append([e["label"], e["games"], e["wins"], e["losses"], e["draws"], fmt_rate(e["win_rate"])])
    return table(["評估對手", "局數", "勝", "敗", "和", "勝率"], rows)


def weight_example_baseline(name: str) -> str:
    path = os.path.join(ROOT, "data", "matches", f"{name}_train.jsonl")
    if not os.path.exists(path):
        return "（找不到棋譜）"
    rec = next(read_matches(path))
    rows = [[st["move_no"], f"({st['action'][0]},{st['action'][1]})", st["deck_size"], st["deck_total"],
             st["weight_before"], f"{st['weight_before'] / st['deck_total'] * 100:.2f}%", st.get("delta"),
             st.get("weight_after")] for st in rec["learning"]]
    return (f"棋局 `{rec['game_id']}`，結果 `{rec['result']}`，共 {rec['move_count']} 手。\n\n"
            + table(["第幾手", "AI 選擇", "可選位置數", "牌組總數", "選中位置牌數（前）", "抽中機率", "增減",
                     "牌數（後）"], rows))


def weight_example_pattern(name: str) -> tuple[str, dict]:
    """修正後：第一場勝局用到的棋形牌，以及它們在之後的變化。"""
    path = os.path.join(ROOT, "data", "matches", f"{name}_train.jsonl")
    if not os.path.exists(path):
        return "（找不到棋譜）", {}
    first_win = None
    for rec in read_matches(path):
        if rec["result"] == "ai_win":
            first_win = rec
            break
    if first_win is None:
        return "訓練期間沒有勝局。", {}
    rows = []
    for st in first_win["learning"]:
        rows.append([st["move_no"], f"({st['action'][0]},{st['action'][1]})", " / ".join(st["patterns"]),
                     " / ".join(map(str, st["cards_before"])), " / ".join(map(str, st["cards_after"])),
                     f"{st['prob'] * 100:.2f}%"])
    txt = (f"第一場勝局 `{first_win['game_id']}`（共 {first_win['move_count']} 手）。棋形字串為該位置某方向左右各 4 格，"
           "0 空、1 自己、2 對手、3 棋盤外。\n\n"
           + table(["第幾手", "AI 選擇", "四個方向的棋形", "棋形牌數（前）", "棋形牌數（後）", "抽中機率"], rows))
    return txt, first_win


def avoid_trigger_stats(n_games: int = 3000, seed: int = 2026) -> dict:
    """重新模擬「修正前＋對手記憶避讓」，統計避讓實際被觸發的次數（實驗輸出未保存逐步紀錄）。"""
    from agents.heuristic_agent import HeuristicAgent
    from agents.menace_agent import MenaceAgent, MenaceConfig
    from training.trainer import run_games
    ai = MenaceAgent(MenaceConfig(opponent_model="avoid"), seed=seed)
    counts = {"steps": 0, "triggers": 0, "games": 0}

    def on(_row):
        counts["steps"] += len(ai.steps)
        t = sum(st.get("avoided", 0) > 0 for st in ai.steps)
        counts["triggers"] += t
        counts["games"] += t > 0

    run_games(ai, HeuristicAgent("basic", deterministic=True), n_games, base_seed=seed, on_game=on)
    return {**counts, "n_games": n_games, **ai.opponent_memory.stats()}


def cross_eval_table(summaries: dict, names: list[str]) -> str:
    labels = []
    for n in names:
        for e in summaries.get(n, {}).get("final_eval", []):
            if e["label"] not in labels:
                labels.append(e["label"])
    rows = []
    for n in names:
        s = summaries.get(n)
        if not s:
            continue
        evals = {e["label"]: e for e in s.get("final_eval", [])}
        rows.append([GROUP_LABEL.get(s.get("group", n), n)]
                    + [fmt_rate(evals[l]["win_rate"]) if l in evals and "win_rate" in evals[l] else "—" for l in labels])
    return table(["設定（種子 2026）"] + labels, rows)


def board_text(moves: list[dict]) -> str:
    g = replay(moves)
    return "```\n" + g.board.render() + "\n```"


def build_report() -> str:
    _setup_font()
    os.makedirs(OUT, exist_ok=True)
    summaries = load_summaries()
    if not summaries:
        raise SystemExit("找不到實驗結果，請先執行 python main.py experiment configs/experiments.json")
    groups = by_group(summaries)

    figs = {
        "main": "learning_curve.png", "seeds": "learning_curve_seeds.png", "color": "first_second.png",
        "sym": "symmetry.png", "rewards": "reward_variants.png", "length": "game_length.png",
        "opp": "opponent_memory.png",
    }
    chart_main(groups, os.path.join(OUT, figs["main"]))
    chart_seeds(groups, os.path.join(OUT, figs["seeds"]))
    chart_dotplot(groups, ["baseline_first", "improved_first", "baseline_second", "improved_second"],
                  os.path.join(OUT, figs["color"]), "先手與後手")
    chart_dotplot(groups, ["baseline_first", "baseline_symmetry_first", "improved_first", "improved_nosym_first"],
                  os.path.join(OUT, figs["sym"]), "對稱去重")
    chart_dotplot(groups, ["baseline_first", "baseline_oppavoid_first", "improved_first", "improved_oppmem_first"],
                  os.path.join(OUT, figs["opp"]), "對手記憶")
    chart_dotplot(groups, ["improved_first", "improved_noloss_first", "improved_nowin_first",
                           "improved_heavyloss_first", "improved_sum_first", "improved_survival_first"],
                  os.path.join(OUT, figs["rewards"]), "獎懲設計")
    chart_length(groups, ["improved_first", "improved_survival_first", "baseline_first"],
                 os.path.join(OUT, figs["length"]))

    b = summaries.get("baseline_first")
    i = summaries.get("improved_first")
    md = ["# 7×7 五子棋位子牌學習 AI：實驗結果（自動產生）", "",
          "本檔由 `python main.py report` 依 `data/experiments/` 的實驗輸出自動產生，所有數字皆來自實際執行。", "",
          "## 定義", "",
          "- **首次勝利**：訓練過程中 AI 第一次獲勝的局號；「首次勝利前的敗局數」= 首次勝利局號 − 1（期間的和局也計入）。",
          "- **穩定勝利**：最近 50 局訓練勝率第一次達到 50% 以上的局號。",
          "- **評估**：複製一份權重、關閉學習後對弈，不會改動受測的學習資料。",
          "- 訓練對手：確定性規則型對手（能連五就連、對手要連五就擋，否則下最靠近中央的空位）。", ""]

    if b and i:
        md += ["## 1. 修正前 vs 修正後（主要實驗，種子 2026，AI 先手）", "",
               f"![學習曲線]({figs['main']})", "",
               table(["", "修正前（整盤局面位子牌）", "修正後（棋形位子牌）"], [
                   ["訓練局數", b["training"]["games"], i["training"]["games"]],
                   ["訓練勝 / 敗 / 和", f"{b['training']['wins']} / {b['training']['losses']} / {b['training']['draws']}",
                    f"{i['training']['wins']} / {i['training']['losses']} / {i['training']['draws']}"],
                   ["首次勝利（第幾局）", fmt(b["training"]["first_win_game"]), fmt(i["training"]["first_win_game"])],
                   ["首次勝利前的敗局數", fmt(b["training"]["losses_before_first_win"]),
                    fmt(i["training"]["losses_before_first_win"])],
                   ["穩定勝利（第幾局）", fmt(b["training"]["stable_win_game"]), fmt(i["training"]["stable_win_game"])],
                   ["學習資料筆數", b["learning_stats"].get("learned_states"), i["learning_stats"].get("learned_patterns")],
               ]), "",
               "每 500 局一段的訓練勝率：", "",
               table(["局數"] + [x["games"] for x in i["training_segments"]],
                     [["修正前"] + [fmt_rate(x["win_rate"]) for x in b["training_segments"]],
                      ["修正後"] + [fmt_rate(x["win_rate"]) for x in i["training_segments"]]]), "",
               "### 訓練後評估（凍結權重）", "", "修正前：", "", final_eval_table(b), "", "修正後：", "",
               final_eval_table(i), ""]

    md += ["## 2. 多個亂數種子", "", f"![各種子學習曲線]({figs['seeds']})", "",
           seed_table(groups, ["baseline_first", "improved_first"]), ""]

    md += ["## 3. 先後手", "", f"![先後手]({figs['color']})", "",
           group_mean_table(groups, ["baseline_first", "baseline_second", "improved_first", "improved_second"]), "",
           seed_table(groups, ["baseline_second", "improved_second"]), ""]

    md += ["## 4. 對稱去重", "", f"![對稱去重]({figs['sym']})", "",
           group_mean_table(groups, ["baseline_first", "baseline_symmetry_first", "improved_first",
                                     "improved_nosym_first"]), ""]
    rows = []
    for gname in ["baseline_first", "baseline_symmetry_first"]:
        for s in groups.get(gname, []):
            ls = s["learning_stats"]
            rows.append([GROUP_LABEL[gname], s["seed"], ls.get("learned_states"),
                         ls.get("unique_states_after_symmetry", "（已去重）"), ls.get("states_visited_once"),
                         ls.get("states_visited_2plus"), ls.get("states_visited_10plus")])
    md += ["修正前（整盤局面）的局面數量：", "",
           table(["設定", "種子", "記錄的局面數", "若做對稱去重剩下", "只遇過 1 次", "遇過 2 次以上", "遇過 10 次以上"], rows), ""]
    rows = []
    for gname in ["improved_first", "improved_nosym_first"]:
        for s in groups.get(gname, []):
            ls = s["learning_stats"]
            rows.append([GROUP_LABEL[gname], s["seed"], ls.get("learned_patterns"), ls.get("patterns_above_initial"),
                         ls.get("patterns_below_initial")])
    md += ["修正後（棋形）的棋形牌數量：", "",
           table(["設定", "種子", "棋形牌數", "高於初始值", "低於初始值"], rows), ""]

    md += ["## 5. 對手記憶", "", f"![對手記憶]({figs['opp']})", "",
           group_mean_table(groups, ["baseline_first", "baseline_oppavoid_first", "improved_first",
                                     "improved_oppmem_first"]), ""]
    rows = []
    for gname in ["baseline_first", "baseline_oppavoid_first"]:
        for s in groups.get(gname, []):
            ls = s["learning_stats"]
            rows.append([GROUP_LABEL[gname], s["seed"], ls.get("opponent_states"), ls.get("opponent_observations"),
                         ls.get("states_seen_twice_or_more"), fmt_rate(ls.get("consistent_reply_ratio")),
                         ls.get("states_with_opponent_winning_reply")])
    md += ["修正前的對手記憶（以整盤局面記錄）：", "",
           table(["設定", "種子", "記錄的對手局面", "觀察手數", "重複遇到的局面", "其中對手每次下同一點", "對手曾直接五連的局面"],
                 rows), ""]

    av = avoid_trigger_stats()
    md += [f"「避讓」實際觸發次數（以種子 2026 重新模擬 {av['n_games']} 局）：AI 共落子 {av['steps']} 次，"
           f"其中 {av['triggers']} 次有候選位置因對手記憶被降低機率（{av['games']} 局）。"
           f"同期對手記憶記錄了 {av['opponent_states']} 個局面，其中 {av['states_with_opponent_winning_reply']} 個是對手曾直接五連的局面；"
           f"重複遇到的 {av['states_seen_twice_or_more']} 個局面中，對手每次都下同一點的比例為 {fmt_rate(av['consistent_reply_ratio'])}。", ""]

    md += ["## 6. 獎懲設計", "", f"![獎懲設計]({figs['rewards']})", "",
           group_mean_table(groups, ["improved_first", "improved_noloss_first", "improved_nowin_first",
                                     "improved_heavyloss_first", "improved_sum_first", "improved_survival_first"]), "",
           f"![每局手數]({figs['length']})", ""]

    md += ["## 6.1 各設定訓練後對不同對手的評估（凍結權重）", "",
           "評估對手中只有第一個是訓練時用的對手；其他對手用來檢查學到的東西能否推廣。", "",
           cross_eval_table(summaries, ["baseline_first", "improved_first", "improved_oppmem_first",
                                        "improved_nosym_first", "improved_heavyloss_first", "improved_survival_first",
                                        "baseline_second", "improved_second", "baseline_vs_random_first",
                                        "improved_vs_random_first"]), ""]

    md += ["## 7. 對隨機對手", "", group_mean_table(groups, ["baseline_vs_random_first", "improved_vs_random_first"]), ""]
    for gname in ["baseline_vs_random_first", "improved_vs_random_first"]:
        for s in groups.get(gname, []):
            md += [f"{GROUP_LABEL[gname]}，每 500 局一段的訓練勝率：",
                   ", ".join(fmt_rate(x["win_rate"]) for x in s["training_segments"]), ""]

    md += ["## 8. 權重變化實例", "", "### 修正前：第 1 局（每個局面的牌組都是新的）", "",
           weight_example_baseline("baseline_first"), ""]
    txt, first_win = weight_example_pattern("improved_first")
    md += ["### 修正後：第一場勝局的棋形牌變化", "", txt, ""]
    if first_win:
        md += ["第一場勝局的終局盤面（X = 黑/AI 先手，O = 白/對手）：", "", board_text(first_win["moves"]), ""]

    md += ["## 9. 由敗至贏的對弈紀錄索引", ""]
    for name in ["baseline_first", "improved_first", "baseline_second", "improved_second"]:
        s = summaries.get(name)
        if not s:
            continue
        rows = load_train(name)
        wins = [r["game_no"] for r in rows if r["result"] == "ai_win"]
        md += [f"- **{GROUP_LABEL[name]}**：棋譜 `{s['files']['train_jsonl']}`；勝局共 {len(wins)} 局；"
               f"前 10 場勝局局號：{', '.join(wins[:10]) or '無'}"]
    md += ["", "重播任一局：`python main.py replay data/matches/improved_first_train.jsonl improved_first-train-000123 --step`", ""]

    md += ["## 10. 全部實驗一覽", "",
           table(["實驗", "說明", "種子", "勝率", "首次勝利", "穩定勝利", "用時（秒）"],
                 [[s["name"], s["description"], s["seed"], fmt_rate(s["training"]["win_rate"]),
                   fmt(s["training"]["first_win_game"]), fmt(s["training"]["stable_win_game"]), s["elapsed_sec"]]
                  for s in summaries.values()]), ""]

    path = os.path.join(OUT, "experiment_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return path
