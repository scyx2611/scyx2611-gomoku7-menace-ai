import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_HERE, '..', '..')))
_sys.path.insert(0, _HERE)
import sys, time
pass
from agents.menace_agent import MenaceAgent, MenaceConfig
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
from training.trainer import run_games, Tally
from training.evaluation import evaluate
from core.board import BLACK, WHITE

def trial(opp, n=2000, color=BLACK, cfg=None):
    ai = MenaceAgent(cfg or MenaceConfig(), seed=1)
    t=time.time(); tally=Tally()
    rows = run_games(ai, opp, n, ai_color=color, base_seed=1, tally=tally)
    s=tally.summary()
    seg = [sum(r["result"]=="ai_win" for r in rows[i:i+n//5])/(n//5) for i in range(0,n,n//5)]
    print(opp.name, opp.params(), "color",color, s["wins"],s["losses"],s["draws"],"first",s["first_win_game"],"stable",s["stable_win_game"],"segments",[round(x,2) for x in seg],"states",len(ai.memory), f"{time.time()-t:.1f}s")
    return ai

for opp in [RandomAgent(seed=2), HeuristicAgent("basic", deterministic=True), HeuristicAgent("pattern", deterministic=True)]:
    trial(opp)
