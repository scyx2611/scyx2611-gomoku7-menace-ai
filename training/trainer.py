"""自動對弈：單局流程、訓練迴圈、統計列。"""

from __future__ import annotations

from collections import deque
from typing import Callable

from agents.base_agent import Agent
from agents.menace_agent import MenaceAgent
from agents.gradient_card_agent import GradientCardAgent
from agents.pattern_menace_agent import PatternMenaceAgent
from agents.rule_only_card_agent import RuleOnlyCardAgent
from core.board import BLACK, COLOR_NAME, WHITE
from core.board import IllegalMoveError
from core.game import Game
from training.rewards import AI_LOSS, AI_WIN, DRAW

ROLLING_WINDOW = 50
LEARNING_AGENTS: tuple = (MenaceAgent, PatternMenaceAgent, GradientCardAgent, RuleOnlyCardAgent)
try:  # 第五種模式需要 PyTorch；沒有安裝時其他模式照常運作
    from agents.neural_card_agent import NeuralCardAgent
    LEARNING_AGENTS += (NeuralCardAgent,)
except ImportError:
    NeuralCardAgent = None


def is_learning_agent(agent: Agent) -> bool:
    return isinstance(agent, LEARNING_AGENTS)


def game_seed(base_seed: int, phase: str, index: int) -> int:
    """每局的亂數種子；同一實驗、同一局號必定相同，可重現。"""
    offset = {"train": 0, "eval": 50_000_000, "play": 90_000_000}.get(phase, 70_000_000)
    return (base_seed * 1_000_003 + offset + index) % (2**31 - 1)


def result_for(winner: int | None, ai_color: int) -> str:
    if winner is None:
        return DRAW
    return AI_WIN if winner == ai_color else AI_LOSS


def play_game(ai: Agent, opponent: Agent, ai_color: int = BLACK, seed: int | None = None) -> tuple[Game, dict | None]:
    """進行一局。回傳 (棋局, AI 的學習紀錄)。

    兩個代理都只拿到棋盤副本，無法直接修改進行中的棋局；
    回傳的位置若不合法會丟出 IllegalMoveError（不會被默默忽略）。
    """
    game = Game()
    agents = {ai_color: ai, -ai_color: opponent}
    ai.start_game(ai_color, seed)
    opponent.start_game(-ai_color, None if seed is None else seed ^ 0x2545F491)
    while not game.is_over:
        color = game.to_move
        before = game.board.copy()
        move = agents[color].select_move(before.copy(), color)
        if not (0 <= move < 49) or game.board.cells[move] != 0:
            raise IllegalMoveError(f"{agents[color].name} 回傳非法位置 {move}")
        game.play_index(move)
        ai.observe_move(before, move, color)
        opponent.observe_move(before, move, color)
    info = ai.end_game(game.winner, ai_color)
    opponent.end_game(game.winner, -ai_color)
    return game, info


def build_record(game: Game, info: dict | None, *, game_id: str, experiment: str, phase: str,
                 ai: Agent, opponent: Agent, ai_color: int, seed: int | None,
                 include_learning: bool = True) -> dict:
    result = result_for(game.winner, ai_color)
    rec = {
        "game_id": game_id,
        "experiment": experiment,
        "phase": phase,
        "ai": {"type": ai.kind, "version": ai.version, "name": ai.name},
        "opponent": opponent.describe(),
        "seed": seed,
        "ai_color": COLOR_NAME[ai_color],
        "ai_first": ai_color == BLACK,
        "moves": [{**m.to_dict(), "by": "ai" if m.color == ai_color else "opponent"} for m in game.history],
        "move_count": len(game.history),
        "winner": COLOR_NAME.get(game.winner) if game.winner else None,
        "result": result,
    }
    if is_learning_agent(ai):
        cfg = ai.config.to_dict()
        rec["reward_params"] = cfg["reward"] if "reward" in cfg else {
            k: cfg[k] for k in ("alpha", "gamma", "lam", "beta", "win_return", "loss_return", "draw_return",
                                "initial_weight") if k in cfg}
        rec["ai_options"] = {k: v for k, v in cfg.items() if k != "reward"}
        rec["learning_enabled"] = ai.learning
        if include_learning and info is not None:
            rec["learning"] = info["steps"]
    return rec


class Tally:
    """累計勝負、滾動勝率、首次勝利、穩定勝利。"""

    def __init__(self, window: int = ROLLING_WINDOW, stable_threshold: float = 0.5):
        self.window = window
        self.stable_threshold = stable_threshold
        self.n = self.wins = self.losses = self.draws = 0
        self.recent: deque[str] = deque(maxlen=window)
        self.first_win: int | None = None
        self.stable_win: int | None = None

    def add(self, result: str) -> dict:
        self.n += 1
        self.wins += result == AI_WIN
        self.losses += result == AI_LOSS
        self.draws += result == DRAW
        self.recent.append(result)
        if result == AI_WIN and self.first_win is None:
            self.first_win = self.n
        rolling = sum(r == AI_WIN for r in self.recent) / len(self.recent)
        if (self.stable_win is None and len(self.recent) == self.window
                and rolling >= self.stable_threshold):
            self.stable_win = self.n
        return {
            "cum_wins": self.wins, "cum_losses": self.losses, "cum_draws": self.draws,
            "cum_win_rate": round(self.wins / self.n, 4),
            f"rolling{self.window}_win_rate": round(rolling, 4),
        }

    def summary(self) -> dict:
        return {
            "games": self.n, "wins": self.wins, "losses": self.losses, "draws": self.draws,
            "win_rate": round(self.wins / self.n, 4) if self.n else None,
            "first_win_game": self.first_win,
            "losses_before_first_win": (self.first_win - 1) if self.first_win else None,
            "stable_win_game": self.stable_win,
            "stable_win_definition": f"最近 {self.window} 局勝率首次 >= {self.stable_threshold:.0%} 的局號",
        }


def color_for(ai_color, index: int) -> int:
    """ai_color 可為 BLACK / WHITE，或 "alternate"（奇數局執黑、偶數局執白）。"""
    if ai_color == "alternate":
        return BLACK if index % 2 == 1 else WHITE
    return ai_color


def run_games(ai: Agent, opponent: Agent, n_games: int, *, ai_color: int | str = BLACK, base_seed: int = 0,
              phase: str = "train", experiment: str = "adhoc", logger=None, start_index: int = 1,
              tally: Tally | None = None, include_learning: bool = True,
              on_game: Callable[[dict], None] | None = None) -> list[dict]:
    """連續對弈 n_games 局，回傳每局統計列；有 logger 時寫入完整棋譜。"""
    tally = tally or Tally()
    rows = []
    for k in range(n_games):
        idx = start_index + k
        seed = game_seed(base_seed, phase, idx)
        color = color_for(ai_color, idx)
        game, info = play_game(ai, opponent, color, seed)
        game_id = f"{experiment}-{phase}-{idx:06d}"
        result = result_for(game.winner, color)
        if logger is not None:
            logger.write(build_record(game, info, game_id=game_id, experiment=experiment, phase=phase, ai=ai,
                                      opponent=opponent, ai_color=color, seed=seed,
                                      include_learning=include_learning))
        row = {"game_no": idx, "game_id": game_id, "result": result, "move_count": len(game.history),
               **({"ai_color": COLOR_NAME[color]} if ai_color == "alternate" else {}),
               **({"opponent_used": opponent.current.name} if hasattr(opponent, "current") else {}),
               **tally.add(result)}
        if is_learning_agent(ai):
            row["learned_entries"] = len(ai.memory)
            row["opponent_states"] = len(ai.opponent_memory)
        rows.append(row)
        if on_game:
            on_game(row)
    return rows


__all__ = ["play_game", "run_games", "color_for", "build_record", "Tally", "game_seed", "result_for", "BLACK", "WHITE"]
