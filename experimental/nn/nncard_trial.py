import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_HERE, '..', '..')))
_sys.path.insert(0, _HERE)
import sys, time; pass
import numpy as np, torch
from nncard import Net, NNCardAgent, selfplay_batch, train_step
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.minimax_agent import MinimaxAgent
from training.trainer import run_games, Tally
from training.tactics import tactical_score
from core.board import BLACK, WHITE
lr = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
ent = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
TOTAL = int(sys.argv[3]) if len(sys.argv) > 3 else 150000
torch.manual_seed(7); rng = np.random.default_rng(7)
net = Net(); opt = torch.optim.Adam(net.parameters(), lr=lr); agent = NNCardAgent(net, seed=1)
def tac():
    s = tactical_score(agent); return f"連五 {s['win']['prob']*100:4.1f} 擋四 {s['block']['prob']*100:4.1f} 擋活三 {s['three']['prob']*100:4.1f}"
def ev():
    out = []
    for name, opp, n in [("規則隨機", HeuristicAgent("basic", False, seed=9), 200), ("棋形", HeuristicAgent("pattern", True), 200),
                         ("MM2", MinimaxAgent(2, seed=9), 100), ("隨機", RandomAgent(seed=9), 200)]:
        for c in (BLACK, WHITE):
            t = Tally(); run_games(agent, opp, n, ai_color=c, base_seed=11, phase="eval", tally=t)
            out.append(f"{name}{'黑' if c == BLACK else '白'} {100*t.wins/n:.0f}/{100*t.draws/n:.0f}")
    return " ".join(out)
print(f"== 神經網路 lr {lr} 熵 {ent}", flush=True)
games, t0 = 0, time.time(); marks = [10000, 30000, 60000, 100000, 150000, 200000, 300000]
while games < TOTAL:
    hist, winner, _ = selfplay_batch(net, 64, rng); games += 64
    info = train_step(net, opt, hist, winner, ent=ent, rng=rng)
    if marks and games >= marks[0]:
        marks.pop(0)
        print(f"  {games:>7} 局：{tac()}  熵 {info.get('entropy', 0):.2f}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  評估 勝%/和%：{ev()}", flush=True)
torch.save(net.state_dict(), f_os.path.join(_HERE, "nn_lr{lr}_ent{ent}.pt"))
