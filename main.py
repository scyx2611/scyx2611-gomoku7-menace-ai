"""7×7 五子棋位子牌學習 AI —— 指令列入口。

    python main.py play                       開啟 Pygame 對弈介面
    python main.py play --cli                 文字模式對弈
    python main.py train  ...                 訓練一個學習 AI 並保存
    python main.py evaluate ...               以凍結權重評估
    python main.py experiment configs/experiments.json [--only 名稱 ...] [--parallel 4]
    python main.py report                     由實驗結果產生圖表與 experiment_report.md
    python main.py replay data/matches/xxx.jsonl GAME_ID
    python main.py verify data/matches/xxx.jsonl   重播檔案中每一局並核對結果
    python main.py stats data/models/xxx.json      學習資料統計
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from core.board import BLACK, NAME_COLOR, WHITE  # noqa: E402


def _opponent_spec(name: str) -> dict:
    presets = {
        "rule": {"type": "heuristic", "level": "basic", "deterministic": True, "name": "rule-deterministic"},
        "rule-random": {"type": "heuristic", "level": "basic", "deterministic": False, "name": "rule-stochastic"},
        "pattern": {"type": "heuristic", "level": "pattern", "deterministic": True, "name": "pattern-deterministic"},
        "random": {"type": "random", "name": "random"},
        "minimax": {"type": "minimax", "depth": 2, "name": "minimax-d2"},
        "rapfi": {"type": "rapfi", "max_node": 20000, "name": "rapfi"},
    }
    if name not in presets:
        raise SystemExit(f"未知的對手 {name}，可用：{', '.join(presets)}")
    return presets[name]


def _load_ai(path: str, learning: bool, seed: int | None = None):
    from agents.gradient_card_agent import GradientCardAgent
    from agents.menace_agent import MenaceAgent
    from agents.pattern_menace_agent import PatternMenaceAgent
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cls = {"pattern-menace-weights": PatternMenaceAgent, "card-pg-weights": GradientCardAgent}.get(
        data.get("format"), MenaceAgent)
    return cls.from_dict(data, learning=learning, seed=seed)


def cmd_train(args):
    from training.experiment import make_agent
    from training.trainer import Tally, run_games
    from reporting.logger import MatchLogger
    if args.load:
        ai = _load_ai(args.load, learning=True, seed=args.seed)
    else:
        ai = make_agent({"type": {"card": "card_pg", "pattern": "pattern_menace", "menace": "menace"}[args.agent]},
                        seed=args.seed)
    opp = make_agent(_opponent_spec(args.opponent), seed=args.seed + 1)
    color = NAME_COLOR[args.color]
    tally = Tally()
    logger = MatchLogger(args.log, overwrite=args.force) if args.log else None
    try:
        run_games(ai, opp, args.games, ai_color=color, base_seed=args.seed, experiment="cli-train",
                  logger=logger, tally=tally, start_index=ai.games_trained + 1)
    finally:
        if logger:
            logger.close()
    ai.save(args.out)
    print(json.dumps(tally.summary(), ensure_ascii=False, indent=2))
    print(f"已保存到 {args.out}")


def cmd_evaluate(args):
    from training.evaluation import evaluate
    from training.experiment import make_agent
    ai = _load_ai(args.model, learning=False, seed=args.seed)
    from agents.rapfi_agent import EngineUnavailableError
    for name in args.opponents:
        try:
            opp = make_agent(_opponent_spec(name), seed=args.seed + 1)
        except EngineUnavailableError as e:
            print(f"{name:>12}: 略過（{e}）")
            continue
        res = evaluate(ai, opp, args.games, ai_color=NAME_COLOR[args.color], base_seed=args.seed, label=name)
        print(f"{name:>12}: {res['wins']} 勝 {res['losses']} 敗 {res['draws']} 和  勝率 {res['win_rate']:.3f}")
        if hasattr(opp, "close"):
            opp.close()


def _run_one(args_tuple):
    cfg, force = args_tuple
    from training.experiment import run_experiment
    s = run_experiment(cfg, force=force)
    return cfg["name"], s["training"]


def cmd_experiment(args):
    from training.experiment import load_config
    configs = load_config(args.config)
    if args.only:
        configs = [c for c in configs if c["name"] in args.only]
        missing = set(args.only) - {c["name"] for c in configs}
        if missing:
            raise SystemExit(f"設定檔中找不到：{', '.join(sorted(missing))}")
    jobs = [(c, args.force) for c in configs]
    if args.parallel > 1:
        from multiprocessing import Pool
        with Pool(args.parallel) as pool:
            for name, tr in pool.imap_unordered(_run_one, jobs):
                print(f"== {name}: 勝率 {tr['win_rate']}, 首次勝利第 {tr['first_win_game']} 局, 穩定勝利第 {tr['stable_win_game']} 局")
    else:
        for job in jobs:
            name, tr = _run_one(job)
            print(f"== {name}: 勝率 {tr['win_rate']}, 首次勝利第 {tr['first_win_game']} 局, 穩定勝利第 {tr['stable_win_game']} 局")


def cmd_report(args):
    from reporting.report import build_report
    path = build_report()
    print(f"報告：{path}")


def cmd_replay(args):
    from core.game import replay
    from reporting.logger import find_match
    rec = find_match(args.file, args.game_id)
    if rec is None:
        raise SystemExit("找不到該棋局")
    game = replay(rec["moves"])
    from core.game import Game
    g = Game()
    for m in rec["moves"]:
        g.play(m["row"], m["col"])
        if args.step:
            print(f"第 {m['n']} 手 {m['player']}（{m['by']}）→ ({m['row']}, {m['col']})")
            print(g.board.render(m["row"] * 7 + m["col"]))
            print()
    print(game.board.render())
    print(f"紀錄結果：{rec['result']}（勝方 {rec['winner']}），重播判定：{game.status}，手數 {len(game.history)}")


def cmd_verify(args):
    from core.board import COLOR_NAME
    from core.game import replay
    from reporting.logger import read_matches
    n = bad = 0
    for rec in read_matches(args.file):
        n += 1
        g = replay(rec["moves"])
        winner = COLOR_NAME.get(g.winner) if g.winner else None
        if winner != rec["winner"]:
            bad += 1
            print(f"不一致：{rec['game_id']}")
    print(f"重播 {n} 局，結果不一致 {bad} 局")
    if bad:
        raise SystemExit(1)


def cmd_stats(args):
    from training.experiment import learning_stats
    ai = _load_ai(args.model, learning=False)
    print(json.dumps({"games_trained": ai.games_trained, **learning_stats(ai)}, ensure_ascii=False, indent=2))


def cmd_play(args):
    if args.cli:
        from ui.cli_play import main as cli_main
        cli_main(args)
    else:
        from ui.pygame_ui import main as ui_main
        ui_main(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="7×7 五子棋位子牌學習 AI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("play", help="人機對弈")
    s.add_argument("--cli", action="store_true", help="使用文字模式")
    s.add_argument("--model", default=None, help="學習 AI 權重檔（預設使用第三版 data/models/card_v3_mixed.json）")
    s.add_argument("--opponent", default="learned",
                   help="learned（第三版）/ improved（第二版）/ baseline（第一版）/ rule / pattern / random / minimax / rapfi")
    s.add_argument("--human", default="white", choices=["black", "white"], help="玩家執黑（先手）或執白")
    s.set_defaults(func=cmd_play)

    s = sub.add_parser("train", help="訓練學習 AI")
    s.add_argument("--agent", default="card", choices=["card", "pattern", "menace"],
                   help="card = 第三版（第二次修正）、pattern = 第二版、menace = 第一版")
    s.add_argument("--opponent", default="rule")
    s.add_argument("--games", type=int, default=1000)
    s.add_argument("--color", default="black", choices=["black", "white"])
    s.add_argument("--seed", type=int, default=2026)
    s.add_argument("--load", default=None, help="從既有權重繼續訓練")
    s.add_argument("--out", default="data/models/cli_model.json")
    s.add_argument("--log", default=None, help="棋譜 JSONL 輸出路徑")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("evaluate", help="凍結權重評估")
    s.add_argument("model")
    s.add_argument("--opponents", nargs="+", default=["rule", "pattern", "random"])
    s.add_argument("--games", type=int, default=200)
    s.add_argument("--color", default="black", choices=["black", "white"])
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("experiment", help="依設定檔執行實驗")
    s.add_argument("config")
    s.add_argument("--only", nargs="+")
    s.add_argument("--parallel", type=int, default=1)
    s.add_argument("--force", action="store_true", help="覆蓋同名實驗（會刪除原始紀錄）")
    s.set_defaults(func=cmd_experiment)

    s = sub.add_parser("report", help="產生圖表與實驗報告")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("replay", help="重播一局")
    s.add_argument("file")
    s.add_argument("game_id")
    s.add_argument("--step", action="store_true", help="逐手顯示")
    s.set_defaults(func=cmd_replay)

    s = sub.add_parser("verify", help="重播並核對整個棋譜檔")
    s.add_argument("file")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("stats", help="學習資料統計")
    s.add_argument("model")
    s.set_defaults(func=cmd_stats)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.chdir(ROOT)
    args.func(args)


if __name__ == "__main__":
    main()
