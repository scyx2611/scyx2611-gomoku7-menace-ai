import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_HERE, '..', '..')))
_sys.path.insert(0, _HERE)
import sys, math
pass
from multiprocessing import Pool
from agents.base_agent import Agent
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from training.trainer import run_games, Tally
from core.board import SIZE, BLACK, WHITE
DIRS=((0,1),(1,0),(1,1),(1,-1))
class PM(Agent):
    kind="pm"
    def __init__(s, init=10, win=3, loss=1, minw=1, comb="sum", seed=None, radius=4):
        super().__init__("pm", seed); s.cards={}; s.init=init; s.win=win; s.loss=loss; s.minw=minw; s.comb=comb; s.radius=radius
    def start_game(s, color, seed=None):
        super().start_game(color, seed); s.steps=[]
    def keys(s, cells, a, color):
        r,c=divmod(a,SIZE); out=[]
        for dr,dc in DIRS:
            ch=[]
            for k in range(-s.radius,s.radius+1):
                if k==0: continue
                rr,cc=r+k*dr,c+k*dc
                if 0<=rr<SIZE and 0<=cc<SIZE:
                    v=cells[rr*SIZE+cc]*color; ch.append("0" if v==0 else "1" if v>0 else "2")
                else: ch.append("3")
            t="".join(ch); rt=t[::-1]; out.append(min(t,rt))
        return out
    def select_move(s, board, color):
        legal=board.legal_moves(); ks=[s.keys(board.cells,a,color) for a in legal]
        ws=[]
        for k4 in ks:
            vals=[s.cards.get(k,s.init) for k in k4]
            ws.append(sum(vals) if s.comb=="sum" else math.prod(vals))
        x=s.rng.random()*sum(ws); acc=0; ch=len(legal)-1
        for i,w in enumerate(ws):
            acc+=w
            if x<acc: ch=i;break
        s.steps.append(ks[ch]); return legal[ch]
    def end_game(s, winner, my):
        d = s.win if winner==my else (-s.loss if winner is not None else 1)
        for k4 in s.steps:
            for k in k4: s.cards[k]=max(s.minw, s.cards.get(k,s.init)+d)
N=6000
def job(a):
    lvl,init,win,loss,minw,comb=a
    ai=PM(init,win,loss,minw,comb,seed=1)
    opp=RandomAgent(seed=3) if lvl=="random" else HeuristicAgent(lvl,deterministic=True)
    t=Tally(); rows=run_games(ai,opp,N,base_seed=1,tally=t)
    seg=[round(sum(r["result"]=="ai_win" for r in rows[i:i+1000])/1000,3) for i in range(0,N,1000)]
    s=t.summary()
    return f"{lvl:7} init{init} win{win} loss{loss} min{minw} {comb} first={s['first_win_game']} stable={s['stable_win_game']} seg={seg} cards={len(ai.cards)}"
grid=[(l,10,3,1,1,c) for l in ["random","basic","pattern"] for c in ["sum","prod"]]+[(l,100,3,1,1,c) for l in ["basic","pattern"] for c in ["sum","prod"]]
with Pool(4) as p:
    for line in p.imap(job,grid): print(line, flush=True)
