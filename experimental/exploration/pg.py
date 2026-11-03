import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_HERE, '..', '..')))
_sys.path.insert(0, _HERE)
import sys, time, math; pass
from multiprocessing import Pool
from agents.base_agent import Agent, weighted_choice
from agents.pattern_menace_agent import window_keys
from training.experiment import make_agent
from agents.mixed_opponent import MixedOpponent
from training.trainer import run_games, Tally
from training.tactics import tactical_score
from core.board import BLACK, WHITE

class PG(Agent):
    """牌數 = exp(θ)。P(a) ∝ Π_d 牌[key_d(a)]。終局後 θ 依 REINFORCE 調整。"""
    kind="pg"
    def __init__(self, alpha=0.1, gamma=0.9, seed=None, shared=None):
        super().__init__("pg", seed); self.alpha=alpha; self.gamma=gamma
        self.theta = shared if shared is not None else {}
        self.learning=True; self.steps=[]
    def twin(self):
        o=PG(self.alpha,self.gamma,shared=self.theta); o.learning=self.learning; return o
    def start_game(self,color,seed=None):
        super().start_game(color,seed); self.steps=[]
    def _dist(self, board, color):
        legal=board.legal_moves(); ks=[window_keys(board.cells,a,color) for a in legal]
        logits=[sum(self.theta.get(k,0.0) for k in kk) for kk in ks]
        m=max(logits); ex=[math.exp(l-m) for l in logits]; z=sum(ex)
        return legal, ks, [e/z for e in ex]
    def move_probabilities(self, board, color):
        legal,_,p=self._dist(board,color); return dict(zip(legal,p))
    def select_move(self, board, color):
        legal,ks,p=self._dist(board,color)
        i=weighted_choice(self.rng,p)
        if self.learning:
            exp={}
            for kk,pi in zip(ks,p):
                for k in kk: exp[k]=exp.get(k,0.0)+pi
            self.steps.append((ks[i],exp))
        return legal[i]
    def end_game(self, winner, my):
        if not self.learning: return None
        G = 0.0 if winner is None else (1.0 if winner==my else -1.0)
        n=len(self.steps)
        for t,(chosen,exp) in enumerate(self.steps):
            g=G*self.gamma**(n-1-t)*self.alpha
            for k,v in exp.items(): self.theta[k]=self.theta.get(k,0.0)-g*v
            for k in chosen: self.theta[k]=self.theta.get(k,0.0)+g
        return None

def pool_for(ai, kind):
    from agents.heuristic_agent import HeuristicAgent
    from agents.random_agent import RandomAgent
    if kind=="mixed":
        return MixedOpponent([(ai.twin(),0.5),(HeuristicAgent("basic",False,seed=3),0.2),(HeuristicAgent("pattern",False,0.2,seed=4),0.2),(RandomAgent(seed=5),0.1)],seed=6)
    if kind=="self": return ai.twin()
    if kind=="rule": return HeuristicAgent("basic",True)
def evals(ai):
    from agents.heuristic_agent import HeuristicAgent
    from agents.random_agent import RandomAgent
    ai.learning=False; out=[]
    for name,opp in [("規則隨機",HeuristicAgent("basic",False,seed=9)),("規則確定",HeuristicAgent("basic",True)),("棋形",HeuristicAgent("pattern",True)),("棋形隨機",HeuristicAgent("pattern",False,seed=9)),("隨機",RandomAgent(seed=9))]:
        for c in (BLACK,WHITE):
            t=Tally(); run_games(ai,opp,200,ai_color=c,base_seed=11,phase="eval",tally=t); out.append(f"{name}{'黑' if c==BLACK else '白'} {t.wins/2:4.1f}/{t.draws/2:4.1f}")
    return " ".join(out)
N=20000
def job(a):
    name,alpha,gamma,kind=a
    ai=PG(alpha,gamma,seed=7); opp=pool_for(ai,kind)
    t=time.time(); run_games(ai,opp,N,ai_color="alternate",base_seed=7)
    tac=tactical_score(ai)
    return f"{name:22} 連五 {tac['win']['prob']*100:5.1f} 擋四 {tac['block']['prob']*100:5.1f} 擋活三 {tac['three']['prob']*100:5.1f} | 勝%/和% {evals(ai)} | 牌 {len(ai.theta)} | {time.time()-t:.0f}s"
jobs=[("PG 混合 α0.1 γ0.9",0.1,0.9,"mixed"),("PG 混合 α0.3 γ0.9",0.3,0.9,"mixed"),("PG 純自我對弈 α0.1 γ0.9",0.1,0.9,"self"),("PG 混合 α0.1 γ1.0",0.1,1.0,"mixed")]
with Pool(4) as p:
    for l in p.imap(job,jobs): print(l, flush=True)
