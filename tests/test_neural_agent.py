"""第五種模式（神經網路）的測試。沒有安裝 PyTorch 時整個檔案略過。"""

import json
import os

import pytest

torch = pytest.importorskip("torch")

from agents.neural_card_agent import NeuralCardAgent, NeuralCardConfig  # noqa: E402
from agents.random_agent import RandomAgent  # noqa: E402
from core.board import BLACK, NUM_CELLS, WHITE, Board  # noqa: E402
from core.game import replay  # noqa: E402
from reporting.logger import MatchLogger, read_matches  # noqa: E402
from training.evaluation import evaluate, fingerprint  # noqa: E402
from training.trainer import Tally, run_games  # noqa: E402

SMALL = NeuralCardConfig(channels=8, blocks=1, batch_games=8)


def test_plays_legal_moves_both_colors_and_learns_per_game():
    ai = NeuralCardAgent(SMALL, seed=1)
    before = fingerprint(ai)
    for color in (BLACK, WHITE):
        rows = run_games(ai, RandomAgent(seed=2), 5, ai_color=color, base_seed=3)
        assert len(rows) == 5
    assert fingerprint(ai) != before   # 有學習


def test_untrained_has_no_win_shortcut_bias():
    # 沒有「會不會連五」的捷徑：連五的位置不會因為規則而被特別挑出（機率接近其他位置）
    ai = NeuralCardAgent(SMALL, seed=1)
    b = Board()
    for c in range(4):
        b.place(3, c, BLACK)
        b.place(5, c, WHITE)
    p = ai.move_probabilities(b, BLACK)
    assert p[3 * 7 + 4] < 0.5


def test_save_load_round_trip(tmp_path):
    ai = NeuralCardAgent(SMALL, seed=1)
    ai.train_games(16, base_seed=1, start_index=1)
    path = os.path.join(tmp_path, "m.pt")
    ai.save(path)
    loaded = NeuralCardAgent.load(path)
    b = Board()
    b.place(3, 3, WHITE)
    a, c = ai.move_probabilities(b, BLACK), loaded.move_probabilities(b, BLACK)
    assert max(abs(a[k] - c[k]) for k in a) < 1e-9
    assert loaded.games_trained == 16


def test_evaluation_does_not_change_weights():
    ai = NeuralCardAgent(SMALL, seed=1)
    before = fingerprint(ai)
    evaluate(ai, RandomAgent(seed=1), 6, base_seed=2)
    assert fingerprint(ai) == before


def test_learning_moves_value_toward_result():
    ai = NeuralCardAgent(SMALL, seed=4)
    rel = [0] * NUM_CELLS
    rel[24] = 1
    import numpy as np
    x = np.array([rel], dtype=np.int8)
    v0 = ai.values(x)[0]
    for _ in range(20):
        ai._fit(np.repeat(x, 8, 0), np.ones(8), np.random.default_rng(0))
    assert ai.values(x)[0] > v0


def test_batched_training_is_reproducible_and_logs_replayable_games(tmp_path):
    def run(path):
        ai = NeuralCardAgent(SMALL, seed=5)
        tally = Tally()
        with MatchLogger(path) as log:
            rows = ai.train_games(20, base_seed=7, start_index=1, logger=log, experiment="t",
                                  include_learning=True, tally=tally)
        return ai, rows
    p1, p2 = os.path.join(tmp_path, "a.jsonl"), os.path.join(tmp_path, "b.jsonl")
    a1, r1 = run(p1)
    a2, r2 = run(p2)
    assert [r["result"] for r in r1] == [r["result"] for r in r2]
    assert fingerprint(a1) == fingerprint(a2)
    recs = list(read_matches(p1))
    assert len(recs) == 20 and {r["opponent"]["type"] for r in recs} <= {"self_play", "random"}
    for rec in recs:
        g = replay(rec["moves"])
        assert ({1: "black", -1: "white"}.get(g.winner)) == rec["winner"]
        assert all("card_after" in st and "target" in st for st in rec["learning"])
    json.dumps(recs[0])
