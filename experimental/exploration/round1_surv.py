import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_HERE, '..', '..')))
_sys.path.insert(0, _HERE)
import sys; pass
from multiprocessing import Pool
from agents.pattern_menace_agent import PatternMenaceAgent, PatternMenaceConfig
from agents.menace_agent import MenaceAgent, MenaceConfig
from agents.heuristic_agent import HeuristicAgent
from training.trainer import run_games, Tally
from training.rewards import AI_LOSS
import statistics
N=10000
class Surv:
    """loss: +s if longer than EMA of loss length else -p"""
    def __init__(s, agent, plus=1, minus=1): s.a=agent; s.ema=None; s.plus=plus; s.minus=minus
def patch(agent, plus, minus):
    rc=agent.config.reward; orig=rc.deltas; st={"ema":None}
    def deltas(result, n):
        if result!=AI_LOSS: return orig(result,n)
        L=2*n  # approx game length (AI moves*2)
        ema=st["ema"]; st["ema"]=L if ema is None else 0.99*ema+0.01*L
        if ema is None: return [0]*n
        return [plus]*n if L>ema else ([-minus]*n if L<ema else [0]*n)
    rc.deltas=deltas
def job(a):
    kind, mode, color = a
    ai = PatternMenaceAgent(seed=1) if kind=="pattern" else MenaceAgent(seed=1)
    if mode=="surv": patch(ai,1,1)
    opp=HeuristicAgent("basic",deterministic=True)
    t=Tally(); rows=run_games(ai,opp,N,ai_color=color,base_seed=1,tally=t)
    seg=[round(sum(r["result"]=="ai_win" for r in rows[i:i+1000])/1000,3) for i in range(0,N,1000)]
    ln=[round(statistics.mean(r["move_count"] for r in rows[i:i+1000]),1) for i in range(0,N,1000)]
    s=t.summary()
    return f"{kind} {mode} color{color} first={s['first_win_game']} stable={s['stable_win_game']} win={seg} len={ln}"
with Pool(4) as p:
    for l in p.imap(job,[("pattern","fixed",1),("pattern","surv",1),("menace","surv",1),("pattern","surv",-1)]): print(l,flush=True)
