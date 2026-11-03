"""原型：位子牌＋神經網路（只靠規則）。

輸入：自己的子、對手的子、全 1 平面（3×7×7）。輸出：每個位置的牌數（logit）與局面勝率估值。
抽籤：牌數 = 10 × e^logit，已有子的位置不可選，依比例抽。
學習（終局後）：G_t = R × γ^(n-1-t)；優勢 A = G_t − V(s_t)
    策略：−log π(a|s) × A（贏了抽中的牌按比例增加，輸了減少）
    估值：(V(s) − G_t)²；另加少量熵獎勵避免太早固定下法。
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_HERE, '..', '..')))
_sys.path.insert(0, _HERE)
import sys, time, math, random
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
INV8 = torch.tensor(INV_PERM)   # 轉換後位置 j ← 原位置 INV[t][j]


class Net(nn.Module):
    def __init__(self, ch=32, blocks=4):
        super().__init__()
        self.inp = nn.Conv2d(3, ch, 3, padding=1)
        self.body = nn.ModuleList([nn.Conv2d(ch, ch, 3, padding=1) for _ in range(blocks)])
        self.pol = nn.Conv2d(ch, 1, 1)
        self.val1 = nn.Linear(ch * N, 64)
        self.val2 = nn.Linear(64, 1)

    def forward(self, x):
        h = F.relu(self.inp(x))
        for conv in self.body:
            h = F.relu(h + conv(h))
        logits = self.pol(h).flatten(1)
        v = torch.tanh(self.val2(F.relu(self.val1(h.flatten(1))))).squeeze(1)
        return logits, v


def encode(cells: np.ndarray, color: np.ndarray) -> torch.Tensor:
    """cells: (B,49) 絕對顏色；color: (B,) 輪到的一方。"""
    rel = cells * color[:, None]
    own = (rel == 1).astype(np.float32); opp = (rel == -1).astype(np.float32)
    ones = np.ones_like(own)
    return torch.from_numpy(np.stack([own, opp, ones], 1).reshape(-1, 3, SIZE, SIZE))


class NNCardAgent(Agent):
    kind = "nn_card"

    def __init__(self, net: Net, seed=None):
        super().__init__("nn-card", seed); self.net = net

    def move_probabilities(self, board, color):
        cells = np.array(board.cells, dtype=np.int8)[None]
        with torch.no_grad():
            logits, _ = self.net(encode(cells, np.array([color])))
        logits = logits[0].numpy().astype(np.float64)
        legal = board.legal_moves()
        z = logits[legal]; p = np.exp(z - z.max()); p /= p.sum()
        return dict(zip(legal, p.tolist()))

    def select_move(self, board, color):
        probs = self.move_probabilities(board, color)
        keys = list(probs); i = weighted_choice(self.rng, [probs[k] for k in keys])
        return keys[i]


def selfplay_batch(net, B, rng, random_frac=0.1):
    """同時下 B 局。回傳訓練樣本 (盤面編碼, 動作, 回報) 與統計。"""
    cells = np.zeros((B, N), dtype=np.int8)
    to_move = np.ones(B, dtype=np.int8)
    done = np.zeros(B, bool); winner = np.zeros(B, dtype=np.int8)
    # 每局有 random_frac 的機率是「網路 vs 隨機」，隨機方是黑或白
    rand_side = np.where(rng.random(B) < random_frac, rng.choice([1, -1], B), 0).astype(np.int8)
    hist = [[] for _ in range(B)]   # (cells copy, color, action)
    moves = 0
    while not done.all():
        act = np.nonzero(~done)[0]
        x = encode(cells[act], to_move[act])
        with torch.no_grad():
            logits, _ = net(x)
        logits = logits.numpy().astype(np.float64)
        for k, g in enumerate(act):
            legal = np.flatnonzero(cells[g] == 0)
            if rand_side[g] == to_move[g]:
                a = int(rng.choice(legal)); learn = False
            else:
                z = logits[k, legal]; p = np.exp(z - z.max()); p /= p.sum()
                a = int(legal[rng.choice(len(legal), p=p)]); learn = True
            if learn:
                hist[g].append((cells[g].copy(), int(to_move[g]), a))
            cells[g, a] = to_move[g]
            r, c = divmod(a, SIZE)
            b = Board(cells[g].tolist())
            if is_winning_move(b, r, c, int(to_move[g])):
                done[g] = True; winner[g] = to_move[g]
            elif not (cells[g] == 0).any():
                done[g] = True
            to_move[g] = -to_move[g]
        moves += len(act)
    return hist, winner, moves


def train_step(net, opt, hist, winner, gamma=0.9, ent=0.01, vcoef=0.5, rng=None):
    X, A, G = [], [], []
    for g, h in enumerate(hist):
        # 每位玩家各自計算回報
        by_color = {1: [], -1: []}
        for st in h:
            by_color[st[1]].append(st)
        for color, steps in by_color.items():
            R = 0.0 if winner[g] == 0 else (1.0 if winner[g] == color else -1.0)
            n = len(steps)
            for t, (cells, col, a) in enumerate(steps):
                X.append((cells, col)); A.append(a); G.append(R * gamma ** (n - 1 - t))
    if not X:
        return {}
    cells = np.stack([c for c, _ in X]); col = np.array([c for _, c in X], dtype=np.int8)
    A = np.array(A); G = torch.tensor(G, dtype=torch.float32)
    # 資料增強：每個樣本隨機套一種旋轉/鏡像（只用到棋盤幾何）
    t = rng.integers(0, 8, len(A))
    perm = np.array(PERM); inv = np.array(INV_PERM)
    cells_t = np.take_along_axis(cells, inv[t], axis=1)
    A_t = perm[t, A]
    x = encode(cells_t, col)
    logits, v = net(x)
    legal = torch.from_numpy(cells_t == 0)
    logits = logits.masked_fill(~legal, -1e9)
    logp = F.log_softmax(logits, 1)
    lp_a = logp[torch.arange(len(A_t)), torch.from_numpy(A_t)]
    adv = (G - v).detach()
    p = logp.exp()
    entropy = -(p * logp.masked_fill(~legal, 0)).sum(1).mean()
    loss = -(lp_a * adv).mean() + vcoef * ((v - G) ** 2).mean() - ent * entropy
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    return {"loss": float(loss.detach()), "entropy": float(entropy.detach()), "n": len(A_t)}


if __name__ == "__main__" and "--speed" in sys.argv:
    torch.manual_seed(0); rng = np.random.default_rng(0)
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    t0 = time.time(); games = 0
    for it in range(10):
        hist, winner, _ = selfplay_batch(net, 64, rng); games += 64
        info = train_step(net, opt, hist, winner, rng=rng)
    print(f"{games} 局 {time.time()-t0:.1f}s", info)
