"""原型：純位子牌（只靠規則）。牌綁在「周圍格子的原始內容」：單格與兩格組合，不分方向。"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_HERE, '..', '..')))
_sys.path.insert(0, _HERE)
import sys, math, time, itertools
pass
import numpy as np
from agents.base_agent import Agent, weighted_choice
from core.board import SIZE, BLACK, WHITE, Board

R = 4
OFFS = [(dr, dc) for dr in range(-R, R + 1) for dc in range(-R, R + 1) if (dr, dc) != (0, 0)]
NO = len(OFFS)
SYM = [lambda r, c: (r, c), lambda r, c: (c, -r), lambda r, c: (-r, -c), lambda r, c: (-c, r),
       lambda r, c: (r, -c), lambda r, c: (-r, c), lambda r, c: (c, r), lambda r, c: (-c, -r)]
OIDX = {o: i for i, o in enumerate(OFFS)}
# 單格：相對位置的對稱類別
single_cls = {}
S_CLASS = []
for o in OFFS:
    key = min(f(*o) for f in SYM)
    S_CLASS.append(single_cls.setdefault(key, len(single_cls)))
NS = len(single_cls)
# 兩格：兩個相對位置（同一個對稱轉換）取標準形，並記錄是否交換順序
pair_cls = {}
PAIRS, P_CLASS, P_SWAP = [], [], []
for i, j in itertools.combinations(range(NO), 2):
    best = None
    for f in SYM:
        a, b = f(*OFFS[i]), f(*OFFS[j])
        cand = (a, b, 0) if a <= b else (b, a, 1)
        if best is None or cand[:2] < best[:2]:
            best = cand
    PAIRS.append((i, j)); P_CLASS.append(pair_cls.setdefault(best[:2], len(pair_cls))); P_SWAP.append(best[2])
PAIRS = np.array(PAIRS); P_CLASS = np.array(P_CLASS); P_SWAP = np.array(P_SWAP)
NP = len(pair_cls)
NFEAT = NS * 4 + NP * 16
# 每個位置的每個相對位置對應到的格子（-1 = 棋盤外）
NEIGH = np.full((SIZE * SIZE, NO), -1)
for a in range(SIZE * SIZE):
    r, c = divmod(a, SIZE)
    for k, (dr, dc) in enumerate(OFFS):
        rr, cc = r + dr, c + dc
        if 0 <= rr < SIZE and 0 <= cc < SIZE:
            NEIGH[a, k] = rr * SIZE + cc
S_BASE = np.array(S_CLASS) * 4
P_BASE = NS * 4 + P_CLASS * 16

def features(cells, color, legal):
    """回傳 (len(legal), NO + npairs) 的特徵編號矩陣。"""
    rel = np.array(cells) * color          # 1 自己, -1 對手, 0 空
    val = np.where(rel == 1, 1, np.where(rel == -1, 2, 0))
    val = np.append(val, 3)                # 索引 -1 → 棋盤外
    V = val[NEIGH[legal]]                  # (L, NO)
    fs = S_BASE[None, :] + V
    v1, v2 = V[:, PAIRS[:, 0]], V[:, PAIRS[:, 1]]
    first = np.where(P_SWAP[None, :] == 1, v2, v1); second = np.where(P_SWAP[None, :] == 1, v1, v2)
    fp = P_BASE[None, :] + first * 4 + second
    return np.concatenate([fs, fp], axis=1)

class RawCard(Agent):
    kind = "raw_card"
    def __init__(self, alpha=0.05, gamma=0.9, seed=None, w=None, pairs=True, norm=False):
        super().__init__("raw-card", seed); self.alpha = alpha; self.gamma = gamma; self.norm = norm
        self.w = w if w is not None else np.zeros(NFEAT); self.learning = True; self.pairs = pairs; self.steps = []
    def twin(self):
        t = RawCard(self.alpha, self.gamma, w=self.w, pairs=self.pairs, norm=self.norm); t.learning = self.learning; return t
    def start_game(self, color, seed=None):
        super().start_game(color, seed); self.steps = []
    def _dist(self, board, color):
        legal = board.legal_moves(); F = features(board.cells, color, legal)
        if not self.pairs: F = F[:, :NO]
        z = self.w[F].sum(axis=1); z -= z.max(); p = np.exp(z); p /= p.sum()
        return legal, F, p
    def move_probabilities(self, board, color):
        legal, _, p = self._dist(board, color); return dict(zip(legal, p.tolist()))
    def select_move(self, board, color):
        legal, F, p = self._dist(board, color)
        i = weighted_choice(self.rng, p.tolist())
        if self.learning: self.steps.append((F, p, i))
        return legal[i]
    def end_game(self, winner, my):
        if not self.learning: return None
        G = 0.0 if winner is None else (1.0 if winner == my else -1.0)
        n = len(self.steps)
        for t, (F, p, i) in enumerate(self.steps):
            k = F.shape[1]
            g = self.alpha * G * self.gamma ** (n - 1 - t) / (k if self.norm else 1)
            if g == 0: continue
            self.w -= g * np.bincount(F.ravel(), weights=np.repeat(p, k), minlength=NFEAT)
            np.add.at(self.w, F[i], g)
        return None

print("相對位置", NO, "單格類別", NS, "兩格類別", NP, "特徵數", NFEAT)
if __name__ == "__main__" and "--speed" in sys.argv:
    from training.trainer import run_games
    ai = RawCard(seed=1); t = time.time(); run_games(ai, ai.twin(), 200, ai_color="alternate", base_seed=1)
    print(f"自我對弈 200 局：{time.time()-t:.1f}s")
