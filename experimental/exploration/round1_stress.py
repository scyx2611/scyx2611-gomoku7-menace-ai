import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_HERE, '..', '..')))
_sys.path.insert(0, _HERE)
import sys, time; pass
from multiprocessing import Pool
def job(k):
    from agents.rapfi_agent import RapfiAgent
    from agents.menace_agent import MenaceAgent
    from training.trainer import play_game, game_seed
    from core.board import WHITE, BLACK
    ai = MenaceAgent(learning=False); r = RapfiAgent(max_node=20000)
    res=[]
    for i in range(150):
        g,_=play_game(ai, r, WHITE if (i+k)%2 else BLACK, seed=game_seed(k,"eval",i)); res.append(tuple(m.index for m in g.history))
    r.close(); return res
t=time.time()
with Pool(6) as p: out=p.map(job, [1,2,3,1,2,3])
print("ok", sum(map(len,out)), "games", f"{time.time()-t:.0f}s", "reproducible:", out[0]==out[3] and out[1]==out[4] and out[2]==out[5])
