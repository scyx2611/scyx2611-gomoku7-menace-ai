"""原型：方向 1「下完之後的盤面好不好」。

盤面好壞 V(盤面) = tanh(Σ 每個 3×3 小區塊的牌值)，小區塊只看原始內容（自己/對手/空）。
每個空位的位子牌數 = 10 × e^(β·V(下在這裡之後的盤面))，依牌數比例抽籤。
終局後依勝負回頭修正：最後一手的目標 = 勝 +1 / 敗 −1 / 和 0，其餘手的目標 = 自己下一手的盤面估值（TD(λ)）。
AI 選位置時不使用「會不會連五」的判斷，連五也要自己學。
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_HERE, '..', '..')))
_sys.path.insert(0, _HERE)
import sys, math
pass
import numpy as np
from agents.base_agent import Agent, weighted_choice
from core.board import SIZE, Board

N = SIZE * SIZE
SYM = [lambda r, c: (r, c), lambda r, c: (c, 6 - r), lambda r, c: (6 - r, 6 - c), lambda r, c: (6 - c, r),
       lambda r, c: (r, 6 - c), lambda r, c: (6 - r, c), lambda r, c: (c, r), lambda r, c: (6 - c, 6 - r)]
SHAPES = [(3, 3)]
if "--rect" in sys.argv:
    SHAPES = [(3, 3), (2, 4), (4, 2)]


def build(shapes):
    """把棋盤上所有指定形狀的小區塊依八種對稱分類。同類小區塊共用同一套牌。"""
    windows = []
    for h, w in shapes:
        for r in range(SIZE - h + 1):
            for c in range(SIZE - w + 1):
                windows.append(tuple((r + i) * SIZE + (c + j) for i in range(h) for j in range(w)))
    classes, class_of, cells, powk, size = {}, [], [], [], []
    for win in windows:
        best = None
        for f in SYM:
            mapped = {cell: f(*divmod(cell, SIZE)) for cell in win}
            key = tuple(sorted(r * SIZE + c for r, c in mapped.values()))
            if best is None or key < best[0]:
                best = (key, mapped)
        key, mapped = best
        cid = classes.setdefault(key, len(classes))
        order = {cell: i for i, cell in enumerate(key)}
        class_of.append(cid)
        cells.append(list(win))
        powk.append([3 ** order[r * SIZE + c] for r, c in (mapped[cell] for cell in win)])
    return classes, class_of, cells, powk


classes, CLASS_OF, CELLS_L, POWK_L = build(SHAPES)
NW, NCLS = len(CELLS_L), len(classes)
CSIZE = [len(k) for k in classes]
OFFSET = np.cumsum([0] + [3 ** n for n in CSIZE])
NFEAT = int(OFFSET[-1])
MAXK = max(len(c) for c in CELLS_L)
CELLS = np.array([c + [N] * (MAXK - len(c)) for c in CELLS_L])          # 補位指到虛擬格（值 0）
POWK = np.array([p + [0] * (MAXK - len(p)) for p in POWK_L])
BASE = OFFSET[np.array(CLASS_OF)]
CW = [[] for _ in range(N)]; CP = [[] for _ in range(N)]
for w in range(NW):
    for cell, p in zip(CELLS_L[w], POWK_L[w]):
        CW[cell].append(w); CP[cell].append(p)
MAXC = max(len(x) for x in CW)
CWA = np.full((N, MAXC), NW); CPA = np.zeros((N, MAXC), dtype=np.int64)
for a in range(N):
    CWA[a, :len(CW[a])] = CW[a]; CPA[a, :len(CP[a])] = CP[a]

def window_ids(cells, color):
    rel = np.asarray(cells) * color
    val = np.append(np.where(rel == 1, 1, np.where(rel == -1, 2, 0)), 0)
    return BASE + (val[CELLS] * POWK).sum(axis=1)     # 每個小區塊的特徵編號

from core.symmetry import PERM as _PERM, INV_PERM as _INV
PERM8 = np.array(_PERM); INV8 = np.array(_INV)


class TDCard(Agent):
    """盤面估值取八種旋轉/鏡像的平均，保證對稱的盤面得到相同的牌數。"""
    kind = "td_card"
    def __init__(self, alpha=0.1, lam=0.7, beta=5.0, seed=None, w=None):
        super().__init__("td-card", seed); self.alpha = alpha; self.lam = lam; self.beta = beta
        self.w = w if w is not None else np.zeros(NFEAT + 1); self.learning = True; self.steps = []
    def twin(self):
        t = TDCard(self.alpha, self.lam, self.beta, w=self.w); t.learning = self.learning; return t
    def start_game(self, color, seed=None):
        super().start_game(color, seed); self.steps = []
    def after_values(self, board, color):
        legal = np.array(board.legal_moves())
        rel = np.asarray(board.cells) * color
        val = np.where(rel == 1, 1, np.where(rel == -1, 2, 0))
        z = np.zeros(len(legal)); per_t = []
        for t in range(8):
            vt = np.append(val[INV8[t]], 0)                    # 轉換後的盤面
            ids = BASE + (vt[CELLS] * POWK).sum(axis=1)
            ids_pad = np.append(ids, NFEAT)
            lt = PERM8[t][legal]                               # 候選位置轉換後的位置
            wsel = CWA[lt]; old = ids_pad[wsel]
            new = np.where(wsel < NW, old + CPA[lt], NFEAT)
            z += self.w[ids].sum() + (self.w[new] - self.w[old]).sum(axis=1)
            per_t.append((ids_pad, wsel, new))
        return legal, np.tanh(z / 8), per_t
    def _probs(self, v):
        x = self.beta * (v - v.max()); p = np.exp(x); return p / p.sum()
    def move_probabilities(self, board, color):
        legal, v, _ = self.after_values(board, color)
        return dict(zip(legal.tolist(), self._probs(v).tolist()))
    def select_move(self, board, color):
        legal, v, per_t = self.after_values(board, color)
        i = weighted_choice(self.rng, self._probs(v).tolist())
        if self.learning:
            feats = []
            for ids_pad, wsel, new in per_t:
                after = ids_pad.copy(); m = wsel[i] < NW
                after[wsel[i][m]] = new[i][m]
                feats.append(after[:NW])
            self.steps.append((np.concatenate(feats), v[i]))
        return int(legal[i])
    def end_game(self, winner, my):
        if not self.learning or not self.steps: return None
        R = 0.0 if winner is None else (1.0 if winner == my else -1.0)
        g = R
        for t in range(len(self.steps) - 1, -1, -1):
            ids, v = self.steps[t]
            if t < len(self.steps) - 1:
                g = (1 - self.lam) * self.steps[t + 1][1] + self.lam * g
            cur = math.tanh(self.w[ids].sum() / 8)
            np.add.at(self.w, ids, self.alpha / NW * (g - cur) * (1 - cur * cur))
        return None

print("形狀", SHAPES, "小區塊", NW, "對稱類別", NCLS, "牌數", NFEAT)
if __name__ == "__main__" and "--speed" in sys.argv:
    import time
    from training.trainer import run_games
    ai = TDCard(seed=1); t = time.time(); run_games(ai, ai.twin(), 300, ai_color="alternate", base_seed=1)
    print(f"自我對弈 300 局：{time.time()-t:.1f}s")
