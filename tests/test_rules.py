"""必要測試 1–5、10：連線判定、邊界、重複落子、終局後落子、平局、重播。"""

import pytest

from core.board import BLACK, SIZE, WHITE, Board, IllegalMoveError
from core.game import BLACK_WIN, DRAW, ONGOING, WHITE_WIN, Game, replay
from core.rules import find_winner, is_winning_move


def play_seq(moves):
    g = Game()
    for r, c in moves:
        g.play(r, c)
    return g


# 黑子位置 + 白子填在不會連線的地方
@pytest.mark.parametrize("black", [
    [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4)],          # 水平
    [(0, 6), (1, 6), (2, 6), (3, 6), (4, 6)],          # 垂直（貼右邊界）
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],          # 正斜線 \
    [(2, 6), (3, 5), (4, 4), (5, 3), (6, 2)],          # 反斜線 /（貼角落）
    [(6, 2), (6, 3), (6, 4), (6, 5), (6, 6)],          # 最下排貼右下角
])
def test_five_in_a_row_all_directions(black):
    white_pool = [(r, c) for r in range(SIZE) for c in range(SIZE) if (r, c) not in black]
    # 白子挑分散位置避免白棋先連五
    whites = [p for p in white_pool if (p[0] + 2 * p[1]) % 5 == 0][:4]
    seq = []
    for i in range(5):
        seq.append(black[i])
        if i < 4:
            seq.append(whites[i])
    g = play_seq(seq)
    assert g.status == BLACK_WIN
    assert g.winner == BLACK
    assert find_winner(g.board) == BLACK


def test_four_is_not_win_and_white_can_win():
    g = play_seq([(0, 0), (6, 0), (0, 1), (6, 1), (0, 2), (6, 2), (0, 3), (6, 3), (5, 5), (6, 4)])
    assert g.status == WHITE_WIN


def test_six_in_a_row_counts_as_win_freestyle():
    b = Board()
    for c in [0, 1, 2, 4, 5, 6]:
        b.place(3, c, BLACK)
    b.place(3, 3, BLACK)
    assert is_winning_move(b, 3, 3, BLACK)


def test_boundary_no_index_error():
    # 四個角與每個邊界格都可以落子並判定，不會越界
    for r in range(SIZE):
        for c in range(SIZE):
            if r in (0, SIZE - 1) or c in (0, SIZE - 1):
                b = Board()
                b.place(r, c, WHITE)
                assert not is_winning_move(b, r, c, WHITE)
    g = Game()
    with pytest.raises(IllegalMoveError):
        g.play(7, 0)
    with pytest.raises(IllegalMoveError):
        g.play(-1, 3)


def test_occupied_cell_rejected():
    g = Game()
    g.play(3, 3)
    with pytest.raises(IllegalMoveError):
        g.play(3, 3)
    assert g.to_move == WHITE and len(g.history) == 1


def test_no_moves_after_game_over():
    g = play_seq([(0, 0), (6, 0), (0, 1), (6, 1), (0, 2), (6, 2), (0, 3), (6, 3), (0, 4)])
    assert g.status == BLACK_WIN
    with pytest.raises(IllegalMoveError):
        g.play(5, 5)
    assert g.legal_moves() == []


def draw_sequence():
    """一個填滿且無任何五連的終局盤面（25 黑、24 白）。棋子不會被移除，所以中途也不可能出現五連。"""
    pattern = ["OXOXOXX", "OOXOOXO", "XOXXOXX", "XOXXXOO", "OXOOXOX", "XOXOXOX", "OOOXXOX"]
    xs = [(r, c) for r in range(SIZE) for c in range(SIZE) if pattern[r][c] == "X"]
    os_ = [(r, c) for r in range(SIZE) for c in range(SIZE) if pattern[r][c] == "O"]
    assert len(xs) == 25 and len(os_) == 24, (len(xs), len(os_))
    seq = []
    for i in range(25):
        seq.append(xs[i])
        if i < 24:
            seq.append(os_[i])
    return seq


def test_full_board_draw():
    g = play_seq(draw_sequence())
    assert g.status == DRAW
    assert g.winner is None
    assert g.board.is_full()


def test_replay_reconstructs_game_and_rejects_bad_records():
    g = play_seq([(3, 3), (2, 2), (3, 4), (2, 3), (3, 2), (2, 4), (3, 1), (2, 5), (3, 5)])
    rec = g.moves_as_dicts()
    g2 = replay(rec)
    assert g2.board == g.board and g2.status == g.status == BLACK_WIN
    bad = [dict(m) for m in rec]
    bad[1]["player"] = "black"
    with pytest.raises(ValueError):
        replay(bad)
    dup = rec[:2] + [dict(rec[0], n=3, player="black")]
    with pytest.raises(IllegalMoveError):
        replay(dup)
    assert replay([]).status == ONGOING
