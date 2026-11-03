import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_HERE, '..', '..')))
_sys.path.insert(0, _HERE)
import sys; pass
from agents.gradient_card_agent import GradientCardAgent
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from core.board import Board, BLACK, WHITE
from core.rules import would_win
from training.evaluation import evaluate

ai = GradientCardAgent.load("data/models/card_v3_mixed.json", learning=False)

# 1) tactical sanity: AI has four -> does it complete five? opponent has four -> does it block?
def prob_of(board, color, target):
    p = ai.move_probabilities(board, color); return p.get(target, 0)
b = Board()
for c in range(4): b.place(3, c, BLACK)
for c in range(3): b.place(5, c+1, WHITE)
b.place(0, 6, WHITE)
print("自己有四子，下第五子連五的機率：%.1f%%" % (100*prob_of(b, BLACK, 3*7+4)))
b2 = Board()
for c in range(4): b2.place(2, c+1, WHITE)
for p in [(4,1),(4,3),(5,5)]: b2.place(*p, BLACK)
print("對手有四子，AI 去擋的機率：%.1f%%（兩端 (2,0),(2,5)）" % (100*(prob_of(b2, BLACK, 2*7+0)+prob_of(b2, BLACK, 2*7+5))))
b3 = Board()
for c in range(3): b3.place(3, c+2, WHITE)
for p in [(5,1),(5,5)]: b3.place(*p, BLACK)
print("對手活三，AI 擋兩端的機率：%.1f%%" % (100*(prob_of(b3, BLACK, 3*7+1)+prob_of(b3, BLACK, 3*7+5))))

# 2) how peaked is the AI's deck on the empty board
p = ai.move_probabilities(Board(), BLACK); top = sorted(p.values(), reverse=True)[:3]
print("空盤最高的三個位置機率：", [round(x*100,1) for x in top])
