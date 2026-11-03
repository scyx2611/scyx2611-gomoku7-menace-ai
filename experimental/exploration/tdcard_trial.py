import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_HERE, '..', '..')))
_sys.path.insert(0, _HERE)
import sys, time; pass
from multiprocessing import Pool
from tdcard import TDCard
from agents.mixed_opponent import MixedOpponent
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.minimax_agent import MinimaxAgent
from training.trainer import run_games, Tally
from training.tactics import tactical_score
from core.board import BLACK, WHITE
def ev(ai):
    ai.learning = False; out = []
    for name, opp, n in [("規則隨機", HeuristicAgent("basic", False, seed=9), 200), ("棋形", HeuristicAgent("pattern", True), 200),
                         ("MM2", MinimaxAgent(2, seed=9), 100), ("隨機", RandomAgent(seed=9), 200)]:
        for c in (BLACK, WHITE):
            t = Tally(); run_games(ai, opp, n, ai_color=c, base_seed=11, phase="eval", tally=t)
            out.append(f"{name}{'黑' if c == BLACK else '白'} {100*t.wins/n:.0f}/{100*t.draws/n:.0f}")
    ai.learning = True; return " ".join(out)
def tac(ai):
    s = tactical_score(ai); return f"連五 {s['win']['prob']*100:4.1f} 擋四 {s['block']['prob']*100:4.1f} 擋活三 {s['three']['prob']*100:4.1f}"
def job(a):
    name, alpha, lam, beta, pool = a
    ai = TDCard(alpha=alpha, lam=lam, beta=beta, seed=7)
    opp = MixedOpponent([(ai.twin(), 0.9), (RandomAgent(seed=3), 0.1)], seed=4) if pool == "mix" else ai.twin()
    lines = []; t0 = time.time(); done = 0
    for chunk in (30000, 70000, 100000, 100000):
        run_games(ai, opp, chunk, ai_color="alternate", base_seed=7, start_index=done + 1); done += chunk
        lines.append(f"  {done:>7} 局：{tac(ai)}  ({time.time()-t0:.0f}s)")
    return f"== {name}\n" + "\n".join(lines) + f"\n  評估 勝%/和%：{ev(ai)}"
tag = "3×3＋2×4＋4×2" if "--rect" in sys.argv else "3×3"
jobs = [(f"{tag} β10", 0.1, 0.7, 10.0, "mix"), (f"{tag} β20", 0.1, 0.7, 20.0, "mix")]
with Pool(2) as p:
    for r in p.imap(job, jobs): print(r, flush=True)
