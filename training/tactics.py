"""基本攻防測驗：衡量學習 AI 是否學會「連五、擋四、擋活三」。

測驗題目從實際對局（隨機版規則型 vs 隨機）中取樣，依規則判定正解：
    win    輪到的一方有位置可以直接連五              → 正解：那些位置
    block  自己不能連五，但對手下一手能連五（擋四）  → 正解：對手的連五點
    three  雙方都沒有四，對手有「活三」               → 正解：活三兩端的空位
           （活三：同一直線上連續三子，兩端都空，且再往外至少一端也空）

規則只用來出題與評分，不參與 AI 的學習。
分數：
    prob    AI 依位子牌抽中正解的平均機率（AI 實際下棋就是按這個機率抽）
    top1    牌數最多的位置是正解的比例
"""

from __future__ import annotations

import random

from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from core.board import BLACK, SIZE, Board, in_bounds, to_index
from core.game import Game
from core.rules import DIRECTIONS, would_win

KINDS = ("win", "block", "three")
KIND_LABEL = {"win": "能連五", "block": "必須擋四", "three": "該擋活三"}


def winning_points(board: Board, color: int) -> list[int]:
    return [i for i in board.legal_moves() if would_win(board, i, color)]


def open_three_ends(board: Board, color: int) -> list[int]:
    """color 的活三兩端空位（簡化定義，只看連續三子）。"""
    ends = set()
    cells = board.cells
    for r in range(SIZE):
        for c in range(SIZE):
            for dr, dc in DIRECTIONS:
                pts = [(r + k * dr, c + k * dc) for k in range(3)]
                if not all(in_bounds(*p) and cells[to_index(*p)] == color for p in pts):
                    continue
                before = (r - dr, c - dc)
                after = (r + 3 * dr, c + 3 * dc)
                if not (in_bounds(*before) and in_bounds(*after)):
                    continue
                if cells[to_index(*before)] != 0 or cells[to_index(*after)] != 0:
                    continue
                # 再往外至少一端也空，才有機會變成活四
                b2, a2 = (r - 2 * dr, c - 2 * dc), (r + 4 * dr, c + 4 * dc)
                room = any(in_bounds(*p) and cells[to_index(*p)] == 0 for p in (b2, a2))
                prev = (r - dr, c - dc)
                if in_bounds(*prev) and cells[to_index(*prev)] == color:
                    continue
                if room:
                    ends.add(to_index(*before))
                    ends.add(to_index(*after))
    return sorted(ends)


def classify(board: Board, color: int) -> tuple[str, list[int]] | None:
    wins = winning_points(board, color)
    if wins:
        return "win", wins
    threats = winning_points(board, -color)
    if threats:
        return "block", threats
    ends = open_three_ends(board, -color)
    if ends and not winning_points(board, color):
        return "three", ends
    return None


def build_positions(n_per_kind: int = 200, seed: int = 12345) -> list[dict]:
    """從隨機對局取樣題目，每類 n_per_kind 題（盤面不重複）。"""
    rng = random.Random(seed)
    found: dict[str, list[dict]] = {k: [] for k in KINDS}
    seen = set()
    players = [HeuristicAgent("basic", deterministic=False, epsilon=0.3, seed=seed), RandomAgent(seed=seed + 1)]
    games = 0
    while any(len(v) < n_per_kind for v in found.values()) and games < 200_000:
        games += 1
        g = Game()
        a, b = rng.sample(players, 2)
        a.start_game(BLACK, rng.randrange(2**31))
        b.start_game(-BLACK, rng.randrange(2**31))
        seats = {BLACK: a, -BLACK: b}
        while not g.is_over:
            color = g.to_move
            key = (tuple(g.board.cells), color)
            if key not in seen:
                seen.add(key)
                res = classify(g.board, color)
                if res and len(found[res[0]]) < n_per_kind:
                    found[res[0]].append({"kind": res[0], "cells": list(g.board.cells), "color": color,
                                          "answers": res[1]})
            g.play_index(seats[color].select_move(g.board.copy(), color))
    return [p for k in KINDS for p in found[k]]


_CACHE: dict[tuple[int, int], list[dict]] = {}


def positions(n_per_kind: int = 200, seed: int = 12345) -> list[dict]:
    key = (n_per_kind, seed)
    if key not in _CACHE:
        _CACHE[key] = build_positions(n_per_kind, seed)
    return _CACHE[key]


def tactical_score(agent, n_per_kind: int = 200, seed: int = 12345) -> dict:
    """回傳 {kind: {"n", "prob", "top1"}}，以及全部平均。agent 需提供 move_probabilities。"""
    out = {}
    for kind in KINDS:
        ps = [p for p in positions(n_per_kind, seed) if p["kind"] == kind]
        probs, top1 = [], []
        for p in ps:
            dist = agent.move_probabilities(Board(p["cells"]), p["color"])
            ans = set(p["answers"])
            probs.append(sum(dist.get(a, 0.0) for a in ans))
            best = max(dist.values())
            tops = [a for a, v in dist.items() if v >= best - 1e-12]
            top1.append(sum(a in ans for a in tops) / len(tops))
        out[kind] = {"n": len(ps), "prob": round(sum(probs) / len(probs), 4) if ps else None,
                     "top1": round(sum(top1) / len(top1), 4) if ps else None}
    out["mean_prob"] = round(sum(out[k]["prob"] for k in KINDS) / len(KINDS), 4)
    return out


class UniformBaseline:
    """對照組：所有空位機率相同（完全沒有學習）。"""

    def move_probabilities(self, board: Board, color: int) -> dict[int, float]:
        legal = board.legal_moves()
        return {a: 1 / len(legal) for a in legal}
