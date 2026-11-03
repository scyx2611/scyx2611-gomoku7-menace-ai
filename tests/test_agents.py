"""必要測試 8、9、11、12 以及學習 AI 的其他行為。"""

import json
import os
import random

import pytest

from agents.gradient_card_agent import GradientCardAgent, GradientCardConfig
from agents.heuristic_agent import HeuristicAgent
from agents.menace_agent import MenaceAgent, MenaceConfig
from agents.minimax_agent import MinimaxAgent
from agents.pattern_menace_agent import PatternMenaceAgent, PatternMenaceConfig, pattern_keys
from agents.random_agent import RandomAgent
from agents.rule_only_card_agent import RuleOnlyCardAgent, RuleOnlyConfig
from agents.rapfi_agent import RapfiAgent, rapfi_available
from core.board import BLACK, NUM_CELLS, WHITE, Board
from core.game import Game, replay
from core.symmetry import transform_cells
from training.evaluation import evaluate, fingerprint
from training.memory import PositionCardMemory
from training.rewards import AI_LOSS, AI_WIN, DRAW, RewardConfig
from training.trainer import play_game, run_games


def learning_agents():
    return [
        MenaceAgent(MenaceConfig(), seed=1),
        MenaceAgent(MenaceConfig(use_symmetry=True, opponent_model="avoid"), seed=1),
        PatternMenaceAgent(PatternMenaceConfig(), seed=1),
        PatternMenaceAgent(PatternMenaceConfig(use_symmetry=False, combine="sum", opponent_model="preempt"), seed=1),
        PatternMenaceAgent(PatternMenaceConfig(features="window"), seed=1),
        GradientCardAgent(GradientCardConfig(), seed=1),
        GradientCardAgent(GradientCardConfig(features="line"), seed=1),
        RuleOnlyCardAgent(seed=1),
    ]


# ---- 8. 權重調整後一定還能選到合法位置 ---------------------------------------
def test_min_weight_zero_never_blocks_selection():
    cfg = MenaceConfig(RewardConfig(win_reward=1, loss_penalty=100, initial_weight=1, min_weight=0))
    ai = MenaceAgent(cfg, seed=3)
    opp = HeuristicAgent("basic", deterministic=True)
    run_games(ai, opp, 300, base_seed=5)  # 大量失敗把許多牌數壓到 0
    assert any(w == 0 for d in ai.memory.states.values() for w in d.values())
    # 把空盤所有位置都歸零，仍必須選得出合法位置（觸發重設）
    key, _ = ai.state_key(Board().cells, BLACK)
    for a in range(NUM_CELLS):
        ai.memory.adjust(key, a, -1000)
    ai.start_game(BLACK, 1)
    move = ai.select_move(Board(), BLACK)
    assert 0 <= move < NUM_CELLS
    assert ai.steps[-1].get("deck_reset") is True


def test_pattern_agent_zero_cards_still_selects():
    cfg = PatternMenaceConfig(RewardConfig(initial_weight=1, min_weight=0, loss_penalty=5))
    ai = PatternMenaceAgent(cfg, seed=1)
    for k in {k for a in range(NUM_CELLS) for k in pattern_keys(Board().cells, a, BLACK)}:
        ai.memory.adjust(k, -5)
    ai.start_game(BLACK, 1)
    assert 0 <= ai.select_move(Board(), BLACK) < NUM_CELLS


@pytest.mark.parametrize("agent", learning_agents(), ids=lambda a: f"{a.kind}")
def test_learning_agents_only_play_legal_moves(agent):
    rng = random.Random(0)
    for _ in range(30):
        g = Game()
        agent.start_game(BLACK, rng.randrange(10**6))
        while not g.is_over:
            if g.to_move == BLACK:
                m = agent.select_move(g.board.copy(), BLACK)
            else:
                m = rng.choice(g.legal_moves())
            assert g.board.cells[m] == 0
            before, mover = g.board.copy(), g.to_move
            g.play_index(m)
            agent.observe_move(before, m, mover)
        agent.end_game(g.winner, BLACK)


# ---- 獎懲 -------------------------------------------------------------------
def test_reward_deltas():
    r = RewardConfig(win_reward=3, loss_penalty=1, draw_reward=1, final_loss_penalty=5)
    assert r.deltas(AI_WIN, 3) == [3, 3, 3]
    assert r.deltas(DRAW, 2) == [1, 1]
    assert r.deltas(AI_LOSS, 3) == [-1, -1, -5]


def test_menace_updates_exactly_the_chosen_cards():
    ai = MenaceAgent(MenaceConfig(), seed=7)
    game, info = play_game(ai, RandomAgent(seed=1), BLACK, seed=11)
    delta = {AI_WIN: 3, AI_LOSS: -1, DRAW: 1}[info["result"]]
    for step in info["steps"]:
        assert step["weight_before"] == 10
        assert step["weight_after"] == max(1, 10 + delta)
        r, c = map(int, step["canon_action"].split(","))
        assert ai.memory.weight(step["state"], r * 7 + c) == step["weight_after"]


def test_menace_symmetry_shares_cards_between_equivalent_boards():
    ai = MenaceAgent(MenaceConfig(use_symmetry=True), seed=0)
    b = Board()
    b.place(0, 1, BLACK); b.place(3, 3, WHITE)
    key, t = ai.state_key(b.cells, BLACK)
    for tt in range(8):
        k2, t2 = ai.state_key(transform_cells(b.cells, tt), BLACK)
        assert k2 == key


# ---- 9. 保存 / 讀取 ----------------------------------------------------------
@pytest.mark.parametrize("agent", learning_agents(), ids=lambda a: f"{a.kind}")
def test_save_load_round_trip(agent, tmp_path):
    run_games(agent, RandomAgent(seed=2), 40, base_seed=9)
    path = os.path.join(tmp_path, "w.json")
    agent.save(path)
    loaded = type(agent).load(path)
    assert loaded.to_dict() == agent.to_dict()
    assert fingerprint(loaded) == fingerprint(agent)
    # 讀回後落子機率一致
    b = Board(); b.place(3, 3, WHITE)
    assert loaded.move_probabilities(b, BLACK) == agent.move_probabilities(b, BLACK)
    with open(path, encoding="utf-8") as f:
        json.load(f)


# ---- 11. 先手 / 後手 ---------------------------------------------------------
@pytest.mark.parametrize("color", [BLACK, WHITE])
@pytest.mark.parametrize("make", [lambda: MenaceAgent(seed=1), lambda: PatternMenaceAgent(seed=1),
                                  lambda: GradientCardAgent(seed=1), lambda: RuleOnlyCardAgent(seed=1)])
def test_ai_plays_as_first_or_second(color, make):
    ai = make()
    rows = run_games(ai, HeuristicAgent("pattern"), 10, ai_color=color, base_seed=4)
    assert len(rows) == 10
    if color == WHITE:
        # 後手時 AI 的第一手是全局第 2 手
        assert ai.steps[0]["move_no"] == 2
    else:
        assert ai.steps[0]["move_no"] == 1


# ---- 12. 評估模式不更動權重 ----------------------------------------------------
@pytest.mark.parametrize("agent", learning_agents(), ids=lambda a: f"{a.kind}")
def test_evaluation_does_not_change_weights(agent):
    run_games(agent, RandomAgent(seed=2), 30, base_seed=1)
    before = json.dumps(agent.to_dict(), sort_keys=True)
    evaluate(agent, HeuristicAgent("basic", deterministic=True), 30, base_seed=2)
    assert json.dumps(agent.to_dict(), sort_keys=True) == before
    frozen = agent.clone(learning=False)
    run_games(frozen, RandomAgent(seed=3), 10, base_seed=3)
    assert json.dumps(frozen.to_dict(), sort_keys=True) == before


# ---- 重現性與紀錄 -------------------------------------------------------------
def test_same_seed_same_games():
    a = run_games(PatternMenaceAgent(seed=0), HeuristicAgent("basic", deterministic=True), 20, base_seed=42)
    b = run_games(PatternMenaceAgent(seed=0), HeuristicAgent("basic", deterministic=True), 20, base_seed=42)
    assert [r["result"] for r in a] == [r["result"] for r in b]
    assert [r["move_count"] for r in a] == [r["move_count"] for r in b]


def test_records_replay_to_same_result(tmp_path):
    from reporting.logger import MatchLogger, read_matches
    path = os.path.join(tmp_path, "m.jsonl")
    ai = MenaceAgent(seed=1)
    with MatchLogger(path) as log:
        run_games(ai, HeuristicAgent("pattern"), 15, ai_color=WHITE, base_seed=3, logger=log, experiment="t")
    recs = list(read_matches(path))
    assert len(recs) == 15
    for rec in recs:
        g = replay(rec["moves"])
        assert (g.winner and {1: "black", -1: "white"}[g.winner]) == rec["winner"] or (g.winner is None and rec["winner"] is None)
        ai_color = BLACK if rec["ai_color"] == "black" else WHITE
        exp = "draw" if g.winner is None else ("ai_win" if g.winner == ai_color else "ai_loss")
        assert exp == rec["result"]
        assert len(rec["learning"]) == sum(1 for m in rec["moves"] if m["by"] == "ai")
    with pytest.raises(FileExistsError):
        MatchLogger(path)


def test_opponent_memory_records_moves():
    ai = MenaceAgent(seed=1)
    opp = HeuristicAgent("basic", deterministic=True)
    run_games(ai, opp, 20, base_seed=1)
    st = ai.opponent_memory.stats()
    assert st["opponent_observations"] > 0
    pm = PatternMenaceAgent(seed=1)
    run_games(pm, opp, 20, base_seed=1)
    assert pm.opponent_memory.observations > 0


def test_position_card_memory_sparse_equivalence():
    m = PositionCardMemory(10, 1)
    assert m.deck("k", [0, 1, 2]) == [10, 10, 10]
    assert m.adjust("k", 1, 5) == (10, 15)
    assert m.adjust("k", 1, -5) == (15, 10)
    assert m.states["k"] == {}
    assert m.adjust("k", 2, -100) == (10, 1)


def test_minimax_takes_win_and_blocks():
    b = Board()
    for c in range(4):
        b.place(2, c, BLACK)
    b.place(5, 5, WHITE)
    mm = MinimaxAgent(depth=2)
    mm.start_game(BLACK, 1)
    assert mm.select_move(b, BLACK) == 2 * 7 + 4
    mm.start_game(WHITE, 1)
    assert mm.select_move(b, WHITE) == 2 * 7 + 4


@pytest.mark.skipif(not rapfi_available(), reason="未安裝 Rapfi 執行檔")
def test_rapfi_plays_full_legal_games_on_7x7():
    r = RapfiAgent(max_node=5000)
    try:
        for color in (BLACK, WHITE):
            game, _ = play_game(r, RandomAgent(seed=1), color, seed=3)
            assert game.is_over
            replay(game.moves_as_dicts())
    finally:
        r.close()


def test_survival_reward_mode():
    r = RewardConfig(survival_mode=True, survival_reward=2, loss_penalty=1)
    assert r.deltas(AI_LOSS, 3, game_length=20, reference_length=15.0) == [2, 2, 2]
    assert r.deltas(AI_LOSS, 3, game_length=10, reference_length=15.0) == [-1, -1, -1]
    assert r.deltas(AI_LOSS, 3, game_length=10, reference_length=None) == [0, 0, 0]
    assert r.deltas(AI_WIN, 2, game_length=10, reference_length=15.0) == [3, 3]


def test_loss_length_average_is_tracked_and_saved(tmp_path):
    cfg = PatternMenaceConfig(RewardConfig(survival_mode=True))
    ai = PatternMenaceAgent(cfg, seed=1)
    run_games(ai, HeuristicAgent("basic", deterministic=True), 30, base_seed=1)
    assert ai.loss_length_avg is not None
    path = os.path.join(tmp_path, "s.json")
    ai.save(path)
    assert PatternMenaceAgent.load(path).loss_length_avg == ai.loss_length_avg


def test_gzip_match_log(tmp_path):
    from reporting.logger import MatchLogger, read_matches
    path = os.path.join(tmp_path, "m.jsonl.gz")
    with MatchLogger(path) as log:
        run_games(RandomAgent(seed=1), RandomAgent(seed=2), 5, logger=log)
    recs = list(read_matches(path))
    assert len(recs) == 5 and all(replay(r["moves"]).is_over for r in recs)


# ---- 第二次修正：第三版 AI、自我對弈、混合對手、攻防測驗 -----------------------------
@pytest.mark.parametrize("result,direction", [("win", 1), ("loss", -1)])
def test_card_pg_update_moves_chosen_probability_in_right_direction(result, direction):
    ai = GradientCardAgent(seed=3)
    b = Board()
    b.place(3, 3, WHITE)
    ai.start_game(BLACK, 1)
    move = ai.select_move(b, BLACK)
    before = ai.move_probabilities(b, BLACK)[move]
    ai.end_game(BLACK if result == "win" else WHITE, BLACK)
    after = ai.move_probabilities(b, BLACK)[move]
    assert (after - before) * direction > 0
    assert ai.steps[0]["cards_after"] != ai.steps[0]["cards_before"]


def test_card_pg_draw_changes_nothing():
    ai = GradientCardAgent(seed=3)
    ai.start_game(BLACK, 1)
    ai.select_move(Board(), BLACK)
    ai.end_game(None, BLACK)
    assert all(abs(v) < 1e-12 for v in ai.memory.theta.values())


def test_twin_shares_cards_and_self_play_learns():
    ai = GradientCardAgent(seed=1)
    tw = ai.twin()
    assert tw.memory is ai.memory and tw.opponent_memory is ai.opponent_memory
    run_games(ai, tw, 20, ai_color="alternate", base_seed=2)
    assert len(ai.memory) > 0 and ai.games_trained == 20 and tw.games_trained == 20


def test_mixed_opponent_and_alternating_colors():
    from agents.mixed_opponent import MixedOpponent
    ai = GradientCardAgent(seed=1)
    opp = MixedOpponent([(RandomAgent(seed=1, name="r"), 1.0), (HeuristicAgent("basic", name="h"), 1.0)], seed=5)
    rows = run_games(ai, opp, 40, ai_color="alternate", base_seed=3)
    assert {r["opponent_used"] for r in rows} == {"r", "h"}
    assert [r["ai_color"] for r in rows[:4]] == ["black", "white", "black", "white"]


def test_tactics_benchmark_classifies_and_scores():
    from training.tactics import UniformBaseline, classify, tactical_score
    b = Board()
    for c in range(4):
        b.place(3, c, BLACK)
    for c in range(3):
        b.place(5, c, WHITE)
    assert classify(b, BLACK) == ("win", [3 * 7 + 4])
    assert classify(b, WHITE)[0] == "block"
    b2 = Board()
    for c in (2, 3, 4):
        b2.place(3, c, WHITE)
    b2.place(0, 0, BLACK)
    kind, ans = classify(b2, BLACK)
    assert kind == "three" and set(ans) == {3 * 7 + 1, 3 * 7 + 5}
    s = tactical_score(UniformBaseline(), n_per_kind=20)
    assert all(0.02 < s[k]["prob"] < 0.2 for k in ("win", "block", "three"))


# ---- 第四種模式：只靠規則 ------------------------------------------------------
def _random_weights(agent, scale=0.3):
    import numpy as np
    agent.memory.w[:-1] = np.random.default_rng(0).normal(size=len(agent.memory.w) - 1) * scale


@pytest.mark.parametrize("shapes", [[[3, 3]], [[3, 3], [2, 4], [4, 2]]])
def test_rule_only_is_exactly_symmetric(shapes):
    from core.symmetry import PERM
    ai = RuleOnlyCardAgent(RuleOnlyConfig(shapes=shapes))
    _random_weights(ai)
    rng = random.Random(4)
    cells = [0] * NUM_CELLS
    for i in rng.sample(range(NUM_CELLS), 13):
        cells[i] = rng.choice([BLACK, WHITE])
    p0 = ai.move_probabilities(Board(cells), BLACK)
    for t in range(8):
        pt = ai.move_probabilities(Board(transform_cells(cells, t)), BLACK)
        assert max(abs(p0[a] - pt[PERM[t][a]]) for a in p0) < 1e-12


def test_rule_only_untrained_is_uniform_and_has_no_win_shortcut():
    # 未訓練時所有空位機率相同：即使下一手就能連五，也沒有任何偏好（連五必須自己學）
    ai = RuleOnlyCardAgent()
    b = Board()
    for c in range(4):
        b.place(3, c, BLACK)
    for c in range(4):
        b.place(5, c, WHITE)
    p = ai.move_probabilities(b, BLACK)
    assert max(p.values()) - min(p.values()) < 1e-12


@pytest.mark.parametrize("result,direction", [("win", 1), ("loss", -1)])
def test_rule_only_learning_moves_card_in_right_direction(result, direction):
    ai = RuleOnlyCardAgent(seed=3)
    ai.record_after = True
    b = Board()
    b.place(3, 3, WHITE)
    ai.start_game(BLACK, 1)
    move = ai.select_move(b, BLACK)
    info = ai.end_game(BLACK if result == "win" else WHITE, BLACK)
    st = info["steps"][0]
    assert (st["card_after"] - st["card_before"]) * direction > 0
    assert "_cells" not in st


def test_rule_only_saves_gzip(tmp_path):
    ai = RuleOnlyCardAgent(seed=1)
    run_games(ai, ai.twin(), 10, ai_color="alternate", base_seed=1)
    path = os.path.join(tmp_path, "m.json.gz")
    ai.save(path)
    loaded = RuleOnlyCardAgent.load(path)
    assert (loaded.memory.w == ai.memory.w).all()
