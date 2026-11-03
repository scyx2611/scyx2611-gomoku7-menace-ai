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
POOL={"type":"mixed","pool":[{"type":"self","weight":0.5},
  {"type":"heuristic","level":"basic","deterministic":False,"weight":0.2},
  {"type":"heuristic","level":"pattern","deterministic":False,"epsilon":0.2,"weight":0.2},
  {"type":"random","weight":0.1}]}
RULE={"type":"heuristic","level":"basic","deterministic":True}
RR={"type":"heuristic","level":"basic","deterministic":False}
PAT={"type":"heuristic","level":"pattern","deterministic":True}
N=20000
def job(a):
    name, feats, reward, opp, color = a
    r={"win_reward":3,"loss_penalty":1,"draw_reward":1,"initial_weight":10,"min_weight":1, **reward}
    ai=make_agent({"type":"pattern_menace","config":{"reward":r,"features":feats}}, seed=7)
    o=make_agent(opp, seed=8, ai=ai)
    t=time.time(); run_games(ai,o,N,ai_color=color,base_seed=7)
    tac=tactical_score(ai)
    ev=lambda spec,c: evaluate(ai, make_agent(spec,seed=9), 200, ai_color=c, base_seed=11)["win_rate"]*100
    return (f"{name:32} 連五 {tac['win']['prob']*100:5.1f} 擋四 {tac['block']['prob']*100:5.1f} 擋活三 {tac['three']['prob']*100:5.1f} | "
            f"規則隨機 黑{ev(RR,BLACK):5.1f} 白{ev(RR,WHITE):5.1f} | 規則確定 黑{ev(RULE,BLACK):5.1f} | 棋形 黑{ev(PAT,BLACK):5.1f} 白{ev(PAT,WHITE):5.1f} | 隨機 黑{ev({'type':'random'},BLACK):5.1f} | 牌 {len(ai.memory)} | {time.time()-t:.0f}s")
SYM={"win_reward":1,"loss_penalty":1}
F={"final_win_reward":30,"final_loss_penalty":10}
jobs=[("視窗 混合 勝+1敗-1 關鍵一手",'window',{**SYM,**F},POOL,"alternate"),
      ("視窗 混合 勝+1敗-1 關鍵一手 上限1000",'window',{**SYM,**F,"max_weight":1000},POOL,"alternate"),
      ("視窗 混合 勝+3敗-1 關鍵一手（對照）",'window',F,POOL,"alternate"),
      ("視窗 混合 勝+1敗-1（無關鍵一手）",'window',SYM,POOL,"alternate"),
      ("視窗 純自我對弈 勝+1敗-1 關鍵一手",'window',{**SYM,**F},{"type":"self"},"alternate"),
      ("視窗 只對確定性規則型 勝+3敗-1（舊獎懲）",'window',{},RULE,BLACK),
      ("舊棋形 混合 勝+1敗-1 關鍵一手",'line',{**SYM,**F},POOL,"alternate"),
      ("視窗 混合 勝+1敗-1 關鍵一手 +10/-10",'window',{**SYM,"final_win_reward":10,"final_loss_penalty":10},POOL,"alternate")]
with Pool(4) as p:
    for l in p.imap(job,jobs): print(l, flush=True)
