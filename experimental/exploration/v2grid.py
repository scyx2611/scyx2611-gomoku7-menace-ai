import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_HERE, '..', '..')))
_sys.path.insert(0, _HERE)
import sys, time; pass
from multiprocessing import Pool
from training.experiment import make_agent
from training.trainer import run_games
from training.evaluation import evaluate
from training.tactics import tactical_score
from core.board import BLACK, WHITE
N=20000
POOL={"type":"mixed","pool":[{"type":"self","weight":0.5},
  {"type":"heuristic","level":"basic","deterministic":False,"weight":0.2,"name":"rule-random"},
  {"type":"heuristic","level":"pattern","deterministic":False,"epsilon":0.2,"weight":0.2,"name":"pattern-eps"},
  {"type":"random","weight":0.1,"name":"random"}]}
RULE={"type":"heuristic","level":"basic","deterministic":True}
def job(a):
    name, reward, opp, color = a
    r={"win_reward":3,"loss_penalty":1,"draw_reward":1,"initial_weight":10,"min_weight":1, **reward}
    ai=make_agent({"type":"pattern_menace","config":{"reward":r}}, seed=7)
    o=make_agent(opp, seed=8, ai=ai)
    t=time.time(); run_games(ai,o,N,ai_color=color,base_seed=7)
    tac=tactical_score(ai)
    ev=lambda spec,c: evaluate(ai, make_agent(spec,seed=9), 200, ai_color=c, base_seed=11)["win_rate"]
    rr={"type":"heuristic","level":"basic","deterministic":False}
    res=f"{name:34} 連五 {tac['win']['prob']*100:5.1f} 擋四 {tac['block']['prob']*100:5.1f} 擋活三 {tac['three']['prob']*100:5.1f} | "
    res+=f"規則隨機 黑{ev(rr,BLACK)*100:5.1f} 白{ev(rr,WHITE)*100:5.1f} | 規則確定 黑{ev(RULE,BLACK)*100:5.1f} | 棋形 黑{ev({'type':'heuristic','level':'pattern','deterministic':True},BLACK)*100:5.1f} | 隨機 黑{ev({'type':'random'},BLACK)*100:5.1f} 白{ev({'type':'random'},WHITE)*100:5.1f} | {time.time()-t:.0f}s"
    return res
jobs=[("舊設定（只對確定性規則型、先手）",{},RULE,BLACK),
      ("1 混合對手＋自我對弈、先後手輪流",{},POOL,"alternate"),
      ("1+2 關鍵一手 勝+30 敗-10",{"final_win_reward":30,"final_loss_penalty":10},POOL,"alternate"),
      ("1+2+3 再加牌數上限 100",{"final_win_reward":30,"final_loss_penalty":10,"max_weight":100},POOL,"alternate"),
      ("1+2+3 牌數上限 30",{"final_win_reward":30,"final_loss_penalty":10,"max_weight":30},POOL,"alternate"),
      ("2+3 只對確定性規則型（不混合）",{"final_win_reward":30,"final_loss_penalty":10,"max_weight":100},RULE,BLACK),
      ("1+3（無關鍵一手）上限 100",{"max_weight":100},POOL,"alternate"),
      ("1+2 敗-10 勝+30、上限 300",{"final_win_reward":30,"final_loss_penalty":10,"max_weight":300},POOL,"alternate")]
with Pool(4) as p:
    for l in p.imap(job,jobs): print(l, flush=True)
