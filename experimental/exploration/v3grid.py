import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_HERE, '..', '..')))
_sys.path.insert(0, _HERE)
import sys, time; pass
from multiprocessing import Pool
from agents.pattern_menace_agent import PatternMenaceAgent, PatternMenaceConfig
from training.rewards import RewardConfig
from training.experiment import make_agent
from training.trainer import run_games
from training.evaluation import evaluate
from training.tactics import tactical_score
from core.board import BLACK, WHITE, SIZE
from core.rules import WINDOWS

# 每個位置、每個方向，通過它的長度 5 視窗
DIRS=((0,1),(1,0),(1,1),(1,-1))
def dir_of(w):
    a,b=divmod(w[0],SIZE),divmod(w[1],SIZE); return (b[0]-a[0], b[1]-a[1])
CW=[[[w for w in WINDOWS if i in w and dir_of(w)==d] for d in DIRS] for i in range(SIZE*SIZE)]

class WindowAgent(PatternMenaceAgent):
    counts=False
    def _keys(self, cells, a, color):
        keys=[]
        for ws in CW[a]:
            om, on, dm, dn = -1, 0, -1, 0
            for w in ws:
                own=opp=0
                for j in w:
                    v=cells[j]*color
                    if v>0: own+=1
                    elif v<0: opp+=1
                if opp==0:
                    if own>om: om,on=own,1
                    elif own==om: on+=1
                if own==0:
                    if opp>dm: dm,dn=opp,1
                    elif opp==dm: dn+=1
            keys.append(f"{om},{min(on,2)}|{dm},{min(dn,2)}" if self.counts else f"{om}|{dm}")
        return keys
    def twin(self, name=None):
        o=super().twin(name); o.__class__=type(self); return o
class WindowAgentC(WindowAgent): counts=True

POOL={"type":"mixed","pool":[{"type":"self","weight":0.5},
  {"type":"heuristic","level":"basic","deterministic":False,"weight":0.2},
  {"type":"heuristic","level":"pattern","deterministic":False,"epsilon":0.2,"weight":0.2},
  {"type":"random","weight":0.1}]}
RULE={"type":"heuristic","level":"basic","deterministic":True}
N=20000
def job(a):
    name, cls, reward, opp, color = a
    r={"win_reward":3,"loss_penalty":1,"draw_reward":1,"initial_weight":10,"min_weight":1, **reward}
    ai=cls(PatternMenaceConfig(RewardConfig(**r)), seed=7)
    o=make_agent(opp, seed=8, ai=ai)
    t=time.time(); run_games(ai,o,N,ai_color=color,base_seed=7)
    tac=tactical_score(ai)
    ev=lambda spec,c: evaluate(ai, make_agent(spec,seed=9), 200, ai_color=c, base_seed=11)["win_rate"]
    rr={"type":"heuristic","level":"basic","deterministic":False}
    res=f"{name:30} 連五 {tac['win']['prob']*100:5.1f} 擋四 {tac['block']['prob']*100:5.1f} 擋活三 {tac['three']['prob']*100:5.1f} | "
    res+=f"規則隨機 黑{ev(rr,BLACK)*100:5.1f} 白{ev(rr,WHITE)*100:5.1f} | 規則確定 黑{ev(RULE,BLACK)*100:5.1f} | 棋形 黑{ev({'type':'heuristic','level':'pattern','deterministic':True},BLACK)*100:5.1f} 白{ev({'type':'heuristic','level':'pattern','deterministic':True},WHITE)*100:5.1f} | 隨機 黑{ev({'type':'random'},BLACK)*100:5.1f} | 牌 {len(ai.memory)} | {time.time()-t:.0f}s"
    return res
K={"final_win_reward":30,"final_loss_penalty":10}
jobs=[("視窗計數 混合對手",WindowAgent,{},POOL,"alternate"),
      ("視窗計數 混合＋關鍵一手",WindowAgent,K,POOL,"alternate"),
      ("視窗計數(含活/死) 混合",WindowAgentC,{},POOL,"alternate"),
      ("視窗計數(含活/死) 混合＋關鍵一手",WindowAgentC,K,POOL,"alternate"),
      ("視窗計數 只對確定性規則型",WindowAgent,{},RULE,BLACK),
      ("視窗計數(含活/死) 混合＋關鍵一手 上限100",WindowAgentC,{**K,"max_weight":100},POOL,"alternate"),
      ("視窗計數(含活/死) 只對隨機版規則型",WindowAgentC,K,{"type":"heuristic","level":"basic","deterministic":False},"alternate"),
      ("視窗計數(含活/死) 純自我對弈＋關鍵一手",WindowAgentC,K,{"type":"self"},"alternate")]
with Pool(4) as p:
    for l in p.imap(job,jobs): print(l, flush=True)
