"""文字模式人機對弈（不需要 Pygame）。輸入「列 行」，例如 3 3。"""

from __future__ import annotations

from core.board import BLACK, NAME_COLOR, IllegalMoveError
from core.game import Game

from .common import ai_choices, save_human_game


def main(args) -> None:
    choices = {key: (label, make) for key, label, make in ai_choices(args.model)}
    key = args.opponent if args.opponent in choices else "learned"
    label, make = choices[key]
    ai, note = make()
    human = NAME_COLOR[args.human]
    print(f"對手：{label}（{note}）。你執{'黑（先手）' if human == BLACK else '白（後手）'}。輸入 q 離開。")
    game = Game()
    ai.start_game(-human)
    while not game.is_over:
        print(game.board.render(game.last_move.index if game.last_move else None))
        before = game.board.copy()
        if game.to_move == human:
            text = input("你的落子（列 行）> ").strip()
            if text.lower() in ("q", "quit", "exit"):
                return
            try:
                r, c = (int(x) for x in text.replace(",", " ").split())
                game.play(r, c)
            except (ValueError, IllegalMoveError) as e:
                print(f"無效：{e}")
                continue
            ai.observe_move(before, r * 7 + c, human)
        else:
            mv = ai.select_move(before.copy(), -human)
            game.play_index(mv)
            ai.observe_move(before, mv, -human)
            print(f"AI 落子：{divmod(mv, 7)}")
    print(game.board.render())
    print({"black_win": "黑勝", "white_win": "白勝", "draw": "平局"}[game.status])
    save_human_game(game, human, ai)
    if hasattr(ai, "close"):
        ai.close()
