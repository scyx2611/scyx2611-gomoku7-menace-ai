import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_HERE, '..', '..')))
_sys.path.insert(0, _HERE)
import sys, time; pass
import numpy as np, torch
from nntd import VNet, NNTDAgent, selfplay_batch, train_step
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.minimax_agent import MinimaxAgent
from training.trainer import run_games, Tally
from training.tactics import tactical_score
from core.board import BLACK, WHITE
TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
torch.manual_seed(7); rng = np.random.default_rng(7)
net = VNet(16, 3); net.load_state_dict(torch.load(sys.argv[2])); opt = torch.optim.Adam(net.parameters(), lr=1e-3); agent = NNTDAgent(net, seed=1)
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
print("== 神經網路估值版 16ch×3 β10（接續第 10 萬局）", flush=True)
games, t0 = int(sys.argv[3]), time.time(); marks = [150000, 200000, 250000, 300000]
while games < TOTAL:
    hist, winner = selfplay_batch(net, 64, rng); games += 64
    info = train_step(net, opt, hist, winner, rng)
    if marks and games >= marks[0]:
        marks.pop(0); print(f"  {games:>7} 局：{tac()}  損失 {info['loss']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        torch.save(net.state_dict(), _os.path.join(_HERE, "nntd_cont_latest.pt"))
print(f"  評估 勝%/和%：{ev()}", flush=True)
