import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_HERE, '..', '..')))
_sys.path.insert(0, _HERE)
import sys, itertools
pass
from multiprocessing import Pool
from agents.menace_agent import MenaceAgent, MenaceConfig
from agents.heuristic_agent import HeuristicAgent
from training.rewards import RewardConfig
from training.trainer import run_games, Tally
from core.board import BLACK

N=6000
def job(args):
    opp_level, init, win, loss, flp, minw, sym, om = args
    cfg = MenaceConfig(RewardConfig(win_reward=win, loss_penalty=loss, draw_reward=1, initial_weight=init, min_weight=minw, final_loss_penalty=flp), use_symmetry=sym, opponent_model=om)
    ai = MenaceAgent(cfg, seed=1); opp = HeuristicAgent(opp_level, deterministic=True)
    t=Tally(); rows=run_games(ai, opp, N, base_seed=1, tally=t)
    seg=[round(sum(r["result"]=="ai_win" for r in rows[i:i+1000])/1000,3) for i in range(0,N,1000)]
    s=t.summary()
    return f"{opp_level:7} init{init:<3} win{win:<3} loss{loss} flp{flp} min{minw} sym{int(sym)} om={om:5} first={s['first_win_game']} stable={s['stable_win_game']} seg={seg} states={len(ai.memory)}"
grid=[]
for lvl in ["basic","pattern"]:
    grid += [(lvl,10,3,1,None,1,False,"off"),
             (lvl,10,3,1,None,1,True,"off"),
             (lvl,10,3,1,None,1,False,"avoid"),
             (lvl,10,3,1,10,0,False,"off"),
             (lvl,3,10,1,None,1,False,"off"),
             (lvl,3,10,1,3,0,True,"off"),
             (lvl,3,30,1,3,0,True,"avoid"),
             (lvl,2,20,2,2,0,True,"avoid")]
with Pool(4) as p:
    for line in p.imap(job, grid): print(line, flush=True)
