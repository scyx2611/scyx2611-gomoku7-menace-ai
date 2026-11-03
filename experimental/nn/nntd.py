"""原型：位子牌＋神經網路（估值版，只靠規則）。

網路只看「下完之後的盤面」（剛落子的一方 = 自己、對手、全 1 平面），輸出勝算估值 V ∈ (−1, 1)。
每個空位的牌數 = 10 × e^(β·V(下在這裡之後的盤面))，依比例抽籤（β 固定，永遠保留探索）。
終局後依勝負回頭算 TD(λ) 目標，以均方誤差訓練網路；訓練資料隨機套用八種旋轉/鏡像。
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_HERE, '..', '..')))
_sys.path.insert(0, _HERE)
import sys, time, math
pass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.base_agent import Agent, weighted_choice
from core.board import SIZE, BLACK, WHITE, Board
from core.rules import is_winning_move
from core.symmetry import PERM, INV_PERM

torch.set_num_threads(1)
N = SIZE * SIZE
INV = np.array(INV_PERM)


class VNet(nn.Module):
    def __init__(self, ch=32, blocks=4):
        super().__init__()
        self.inp = nn.Conv2d(3, ch, 3, padding=1)
        self.body = nn.ModuleList([nn.Conv2d(ch, ch, 3, padding=1) for _ in range(blocks)])
        self.v1 = nn.Linear(ch * N, 64)
        self.v2 = nn.Linear(64, 1)

    def forward(self, x):
        h = F.relu(self.inp(x))
        for conv in self.body:
            h = F.relu(h + conv(h))
        return torch.tanh(self.v2(F.relu(self.v1(h.flatten(1))))).squeeze(1)


def planes(rel: np.ndarray) -> torch.Tensor:
    """rel: (B,49)，1 = 自己（剛落子的一方）、-1 = 對手。"""
    own = (rel == 1).astype(np.float32); opp = (rel == -1).astype(np.float32)
    return torch.from_numpy(np.stack([own, opp, np.ones_like(own)], 1).reshape(-1, 3, SIZE, SIZE))


def after_values(net, cells: np.ndarray, color: int):
    """一個盤面所有空位「下完之後」的估值。"""
    legal = np.flatnonzero(cells == 0)
    rel = np.repeat((cells * color)[None], len(legal), 0)
    rel[np.arange(len(legal)), legal] = 1
    with torch.no_grad():
        v = net(planes(rel)).numpy().astype(np.float64)
    return legal, v


class NNTDAgent(Agent):
    kind = "nn_td_card"

    def __init__(self, net, beta=10.0, seed=None):
        super().__init__("nn-td-card", seed); self.net = net; self.beta = beta

    def move_probabilities(self, board, color):
        legal, v = after_values(self.net, np.array(board.cells, dtype=np.int8), color)
        x = self.beta * (v - v.max()); p = np.exp(x); p /= p.sum()
        return dict(zip(legal.tolist(), p.tolist()))

    def select_move(self, board, color):
        probs = self.move_probabilities(board, color)
        keys = list(probs); return keys[weighted_choice(self.rng, [probs[k] for k in keys])]


def selfplay_batch(net, B, rng, beta=10.0, random_frac=0.1):
    cells = np.zeros((B, N), dtype=np.int8)
    to_move = np.ones(B, dtype=np.int8)
    done = np.zeros(B, bool); winner = np.zeros(B, dtype=np.int8)
    rand_side = np.where(rng.random(B) < random_frac, rng.choice([1, -1], B), 0).astype(np.int8)
    hist = [[] for _ in range(B)]          # (下完後的相對盤面, 落子方, 當時估值)
    while not done.all():
        act = np.nonzero(~done)[0]
        # 一次算出所有進行中棋局的所有候選盤面
        rels, owners, legals = [], [], []
        for g in act:
            legal = np.flatnonzero(cells[g] == 0)
            r = np.repeat((cells[g] * to_move[g])[None], len(legal), 0)
            r[np.arange(len(legal)), legal] = 1
            rels.append(r); legals.append(legal); owners.append(len(legal))
        with torch.no_grad():
            v_all = net(planes(np.concatenate(rels))).numpy().astype(np.float64)
        pos = 0
        for k, g in enumerate(act):
            legal = legals[k]; v = v_all[pos:pos + len(legal)]; r = rels[k]; pos += len(legal)
            if rand_side[g] == to_move[g]:
                i = int(rng.integers(len(legal)))
            else:
                x = beta * (v - v.max()); p = np.exp(x); p /= p.sum()
                i = int(rng.choice(len(legal), p=p))
                hist[g].append((r[i].copy(), int(to_move[g]), float(v[i])))
            a = int(legal[i])
            cells[g, a] = to_move[g]
            rr, cc = divmod(a, SIZE)
            if is_winning_move(Board(cells[g].tolist()), rr, cc, int(to_move[g])):
                done[g] = True; winner[g] = to_move[g]
            elif not (cells[g] == 0).any():
                done[g] = True
            to_move[g] = -to_move[g]
    return hist, winner


def train_step(net, opt, hist, winner, rng, lam=0.7):
    X, Y = [], []
    for g, h in enumerate(hist):
        for color in (1, -1):
            steps = [s for s in h if s[1] == color]
            if not steps:
                continue
            target = 0.0 if winner[g] == 0 else (1.0 if winner[g] == color else -1.0)
            for t in range(len(steps) - 1, -1, -1):
                if t < len(steps) - 1:
                    target = (1 - lam) * steps[t + 1][2] + lam * target
                X.append(steps[t][0]); Y.append(target)
    if not X:
        return {}
    X = np.stack(X); tr = rng.integers(0, 8, len(X))
    X = np.take_along_axis(X, INV[tr], axis=1)          # 八種旋轉/鏡像擇一
    v = net(planes(X))
    loss = F.mse_loss(v, torch.tensor(Y, dtype=torch.float32))
    opt.zero_grad(); loss.backward(); opt.step()
    return {"loss": float(loss.detach()), "n": len(Y)}


if __name__ == "__main__" and "--speed" in sys.argv:
    torch.manual_seed(0); rng = np.random.default_rng(0)
    net = VNet(); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    t0 = time.time()
    for _ in range(5):
        hist, winner = selfplay_batch(net, 64, rng)
        info = train_step(net, opt, hist, winner, rng)
    print(f"320 局 {time.time()-t0:.1f}s", info)
