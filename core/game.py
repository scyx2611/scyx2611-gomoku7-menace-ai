"""對弈流程：回合控制、勝負/平局判定、完整落子歷史。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .board import BLACK, COLOR_NAME, WHITE, Board, IllegalMoveError, in_bounds, to_index, to_rc
from .rules import find_winner, is_winning_move

ONGOING = "ongoing"
BLACK_WIN = "black_win"
WHITE_WIN = "white_win"
DRAW = "draw"


@dataclass
class Move:
    number: int  # 第幾手（從 1 開始）
    color: int
    row: int
    col: int

    @property
    def index(self) -> int:
        return to_index(self.row, self.col)

    def to_dict(self) -> dict:
        return {"n": self.number, "player": COLOR_NAME[self.color], "row": self.row, "col": self.col}


@dataclass
class Game:
    """一局五子棋。黑棋（BLACK）永遠先手。"""

    board: Board = field(default_factory=Board)
    to_move: int = BLACK
    status: str = ONGOING
    history: list[Move] = field(default_factory=list)

    @property
    def is_over(self) -> bool:
        return self.status != ONGOING

    @property
    def winner(self) -> int | None:
        return {BLACK_WIN: BLACK, WHITE_WIN: WHITE}.get(self.status)

    @property
    def last_move(self) -> Move | None:
        return self.history[-1] if self.history else None

    def legal_moves(self) -> list[int]:
        return [] if self.is_over else self.board.legal_moves()

    def play(self, row: int, col: int) -> str:
        """目前輪到的一方在 (row, col) 落子，回傳落子後的棋局狀態。"""
        if self.is_over:
            raise IllegalMoveError("棋局已結束，不得繼續落子")
        if not in_bounds(row, col):
            raise IllegalMoveError(f"({row}, {col}) 超出棋盤範圍")
        color = self.to_move
        self.board.place(row, col, color)  # 已佔據會丟出 IllegalMoveError
        self.history.append(Move(len(self.history) + 1, color, row, col))
        if is_winning_move(self.board, row, col, color):
            self.status = BLACK_WIN if color == BLACK else WHITE_WIN
        elif self.board.is_full():
            self.status = DRAW
        else:
            self.to_move = -color
        return self.status

    def play_index(self, idx: int) -> str:
        return self.play(*to_rc(idx))

    def reset(self) -> None:
        self.board.reset()
        self.to_move = BLACK
        self.status = ONGOING
        self.history.clear()

    def moves_as_dicts(self) -> list[dict]:
        return [m.to_dict() for m in self.history]


def replay(moves: list[dict]) -> Game:
    """依照紀錄 [{"row":..,"col":..,"player":..}, ...] 重建整局並驗證。

    - 每一手都必須合法且依黑白交替；
    - 終局狀態由規則重新計算（不是讀取紀錄），呼叫端可再與紀錄的結果比對。
    """
    game = Game()
    for i, m in enumerate(moves):
        expected = COLOR_NAME[game.to_move]
        if "player" in m and m["player"] != expected:
            raise ValueError(f"第 {i + 1} 手應由 {expected} 落子，紀錄為 {m['player']}")
        game.play(m["row"], m["col"])
    # 雙重驗證：整盤掃描結果必須與逐手判定一致
    scanned = find_winner(game.board)
    if scanned != game.winner:
        raise ValueError("重播後整盤掃描的勝負與逐手判定不一致")
    return game
