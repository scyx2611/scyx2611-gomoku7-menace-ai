"""Pygame 對弈介面。

功能：7×7 棋盤、滑鼠落子、AI 落子提示（最後一手標記）、回合顯示、勝負提示、
重新開始、切換 AI 對手、切換執黑/執白、顯示位子牌機率、開始/停止訓練、保存模型、統計。

執行：python main.py play
"""

from __future__ import annotations

import os
import time

import pygame

from agents.heuristic_agent import HeuristicAgent
from core.board import BLACK, NAME_COLOR, SIZE, WHITE, to_index
from core.game import Game
from training.trainer import Tally, is_learning_agent, run_games

from .common import ROOT, ai_choices, save_human_game

W, H = 980, 660
BOARD_PX = 600
MARGIN = 70
CELL = (BOARD_PX - 2 * MARGIN) // (SIZE - 1)
PANEL_X = BOARD_PX + 30

COLORS = {
    "bg": (236, 239, 242), "board": (222, 184, 125), "line": (70, 52, 30), "black": (24, 24, 24),
    "white": (250, 250, 250), "text": (30, 36, 42), "muted": (95, 105, 115), "accent": (31, 111, 120),
    "button": (255, 255, 255), "button_border": (190, 198, 206), "hot": (200, 70, 50), "win": (40, 140, 70),
}
FONT_CANDIDATES = ["microsoftjhenghei", "microsoftjhengheiui", "msjh", "pingfangtc", "notosanscjktc",
                   "notosanstc", "wenquanyizenhei", "wqyzenhei", "simhei", "mingliu", "arialunicodems"]


def load_font(size: int) -> pygame.font.Font:
    for name in FONT_CANDIDATES:
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.Font(None, size)


class Button:
    def __init__(self, key: str, rect: tuple[int, int, int, int]):
        self.key = key
        self.rect = pygame.Rect(rect)


class App:
    def __init__(self, model: str | None = None, opponent: str = "learned", human: str = "white"):
        pygame.init()
        self.font = load_font(20)
        self.small = load_font(15)
        self.big = load_font(28)
        self.choices = ai_choices(model)
        keys = [k for k, _, _ in self.choices]
        self.choice_idx = keys.index(opponent) if opponent in keys else 0
        self.human = NAME_COLOR[human]
        self.show_probs = True
        self.training = False
        self.train_tally = Tally(window=100)
        self.train_opponent = HeuristicAgent("basic", deterministic=True)
        self.message = ""
        self.ai = None
        self._select_ai()
        self.new_game()
        y = 250
        self.buttons = []
        for key in ("restart", "opponent", "color", "probs", "train", "save"):
            self.buttons.append(Button(key, (PANEL_X, y, 320, 40)))
            y += 50

    # ---- 狀態 -------------------------------------------------------------
    def _select_ai(self) -> None:
        if self.ai is not None and hasattr(self.ai, "close"):
            self.ai.close()
        key, label, make = self.choices[self.choice_idx]
        try:
            self.ai, note = make()
        except Exception as e:  # 例如模型檔不存在
            self.ai, note = None, f"無法載入：{e}"
        self.ai_label, self.ai_note = label, note
        self.training = False

    def new_game(self) -> None:
        self.game = Game()
        self.last_ai_probs: dict[int, float] = {}
        self.last_ai_move: int | None = None
        self.saved = False
        self.ai_wait_until = time.time() + 0.4
        if self.ai is not None:
            self.ai.start_game(-self.human)

    @property
    def ai_color(self) -> int:
        return -self.human

    # ---- 互動 -------------------------------------------------------------
    def cell_at(self, pos: tuple[int, int]) -> int | None:
        x, y = pos
        c = round((x - MARGIN) / CELL)
        r = round((y - MARGIN) / CELL)
        if 0 <= r < SIZE and 0 <= c < SIZE:
            cx, cy = MARGIN + c * CELL, MARGIN + r * CELL
            if (x - cx) ** 2 + (y - cy) ** 2 <= (CELL * 0.45) ** 2:
                return to_index(r, c)
        return None

    def play_index(self, idx: int) -> None:
        before = self.game.board.copy()
        color = self.game.to_move
        self.game.play_index(idx)
        if self.ai is not None:
            self.ai.observe_move(before, idx, color)

    def click(self, pos: tuple[int, int]) -> None:
        for b in self.buttons:
            if b.rect.collidepoint(pos):
                self.press(b.key)
                return
        if self.training or self.game.is_over or self.game.to_move != self.human:
            return
        idx = self.cell_at(pos)
        if idx is not None and self.game.board.cells[idx] == 0:
            self.play_index(idx)
            self.ai_wait_until = time.time() + 0.35

    def press(self, key: str) -> None:
        if key == "restart":
            self.new_game()
        elif key == "opponent":
            self.choice_idx = (self.choice_idx + 1) % len(self.choices)
            self._select_ai()
            self.new_game()
        elif key == "color":
            self.human = -self.human
            self.new_game()
        elif key == "probs":
            self.show_probs = not self.show_probs
        elif key == "train":
            if not is_learning_agent(self.ai):
                self.message = "只有位子牌 AI 可以訓練"
                return
            self.training = not self.training
            self.ai.learning = self.training
            if not self.training:
                self.new_game()
        elif key == "save":
            if is_learning_agent(self.ai):
                path = os.path.join(ROOT, "data", "models", f"ui_{self.ai.kind}.json")
                self.ai.save(path)
                self.message = f"已保存 {os.path.relpath(path, ROOT)}"
            else:
                self.message = "此對手沒有學習資料可保存"

    def update(self) -> None:
        if self.training and self.ai is not None:
            # 每個畫面訓練 20 局，介面保持可操作
            run_games(self.ai, self.train_opponent, 20, ai_color=self.ai_color, base_seed=int(time.time()),
                      tally=self.train_tally, start_index=self.train_tally.n + 1, experiment="ui")
            return
        g = self.game
        if g.is_over:
            if not self.saved and self.ai is not None:
                save_human_game(g, self.human, self.ai)
                self.saved = True
            return
        if self.ai is not None and g.to_move == self.ai_color and time.time() >= self.ai_wait_until:
            # 先記下 AI 這一手所抽的那副位子牌，畫面上顯示的就是它實際抽牌時的機率
            if is_learning_agent(self.ai):
                self.last_ai_probs = self.ai.move_probabilities(g.board, self.ai_color)
            idx = self.ai.select_move(g.board.copy(), self.ai_color)
            self.last_ai_move = idx
            self.play_index(idx)

    # ---- 繪圖 -------------------------------------------------------------
    def text(self, surf, s: str, pos, font=None, color=None) -> None:
        surf.blit((font or self.font).render(s, True, color or COLORS["text"]), pos)

    def draw(self, surf: pygame.Surface) -> None:
        surf.fill(COLORS["bg"])
        pygame.draw.rect(surf, COLORS["board"], (10, 10, BOARD_PX - 20, BOARD_PX - 20), border_radius=6)
        for i in range(SIZE):
            p = MARGIN + i * CELL
            pygame.draw.line(surf, COLORS["line"], (MARGIN, p), (MARGIN + (SIZE - 1) * CELL, p), 2)
            pygame.draw.line(surf, COLORS["line"], (p, MARGIN), (p, MARGIN + (SIZE - 1) * CELL), 2)
            self.text(surf, str(i), (p - 5, 20), self.small, COLORS["muted"])
            self.text(surf, str(i), (22, p - 9), self.small, COLORS["muted"])
        c = MARGIN + (SIZE // 2) * CELL
        pygame.draw.circle(surf, COLORS["line"], (c, c), 5)

        probs = self.last_ai_probs if (self.show_probs and not self.training) else {}
        top = max(probs.values()) if probs else 0
        for idx, v in enumerate(self.game.board.cells):
            r, col = divmod(idx, SIZE)
            x, y = MARGIN + col * CELL, MARGIN + r * CELL
            if v != 0:
                pygame.draw.circle(surf, COLORS["black"] if v == BLACK else COLORS["white"], (x, y), CELL // 2 - 6)
                pygame.draw.circle(surf, COLORS["line"], (x, y), CELL // 2 - 6, 1)
                if idx == self.last_ai_move and idx in probs:
                    self.text(surf, f"{probs[idx] * 100:.0f}%", (x + 12, y + 12), self.small, COLORS["hot"])
            elif idx in probs:
                p = probs[idx]
                radius = max(3, int((CELL // 2 - 14) * (p / top) ** 0.5)) if top else 3
                pygame.draw.circle(surf, COLORS["hot"], (x, y), radius, 2)
                if p >= 0.01:
                    self.text(surf, f"{p * 100:.0f}%", (x + 8, y + 6), self.small, COLORS["hot"])
        last = self.game.last_move
        if last:
            x, y = MARGIN + last.col * CELL, MARGIN + last.row * CELL
            pygame.draw.circle(surf, COLORS["hot"], (x, y), 7)

        # ---- 側欄 ----
        self.text(surf, "7×7 五子棋 位子牌 AI", (PANEL_X, 20), self.big)
        self.text(surf, f"對手：{self.ai_label}", (PANEL_X, 70))
        self.text(surf, self.ai_note, (PANEL_X, 98), self.small, COLORS["muted"])
        you = "黑（先手）" if self.human == BLACK else "白（後手）"
        self.text(surf, f"你執{you}", (PANEL_X, 125))
        g = self.game
        if self.training:
            status, color = "訓練中…", COLORS["accent"]
        elif g.is_over:
            if g.winner is None:
                status, color = "平局", COLORS["muted"]
            elif g.winner == self.human:
                status, color = "你贏了！", COLORS["win"]
            else:
                status, color = "AI 獲勝", COLORS["hot"]
        else:
            status = "輪到你下" if g.to_move == self.human else "AI 思考中…"
            color = COLORS["accent"]
        self.text(surf, status, (PANEL_X, 160), self.big, color)
        self.text(surf, f"第 {len(g.history)} 手", (PANEL_X, 200), self.small, COLORS["muted"])
        if probs and self.last_ai_move is not None:
            self.text(surf, f"紅圈：AI 上一手的位子牌機率，抽中 {probs.get(self.last_ai_move, 0) * 100:.1f}%",
                      (PANEL_X, 222), self.small, COLORS["hot"])

        labels = {
            "restart": "重新開始",
            "opponent": "切換對手",
            "color": "改執白（後手）" if self.human == BLACK else "改執黑（先手）",
            "probs": "隱藏位子牌機率" if self.show_probs else "顯示位子牌機率",
            "train": "停止訓練" if self.training else "開始訓練（對規則型）",
            "save": "保存模型",
        }
        for b in self.buttons:
            pygame.draw.rect(surf, COLORS["button"], b.rect, border_radius=6)
            pygame.draw.rect(surf, COLORS["button_border"], b.rect, 1, border_radius=6)
            self.text(surf, labels[b.key], (b.rect.x + 14, b.rect.y + 9))

        y = 560
        if self.train_tally.n:
            t = self.train_tally
            recent = sum(r == "ai_win" for r in t.recent) / len(t.recent)
            self.text(surf, f"本次已訓練 {t.n} 局：{t.wins} 勝 {t.losses} 敗 {t.draws} 和", (PANEL_X, y), self.small)
            self.text(surf, f"最近 {len(t.recent)} 局勝率 {recent:.0%}；首次勝利：第 {t.first_win or '-'} 局",
                      (PANEL_X, y + 22), self.small)
            y += 48
        if is_learning_agent(self.ai):
            self.text(surf, f"學習資料：{len(self.ai.memory)} 筆，累計訓練 {self.ai.games_trained} 局",
                      (PANEL_X, y), self.small, COLORS["muted"])
            y += 22
        if self.message:
            self.text(surf, self.message, (PANEL_X, y), self.small, COLORS["accent"])


def main(args=None) -> None:
    app = App(getattr(args, "model", None), getattr(args, "opponent", "learned"), getattr(args, "human", "white"))
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("7×7 五子棋 位子牌學習 AI")
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                app.click(event.pos)
        app.update()
        app.draw(screen)
        pygame.display.flip()
        clock.tick(30)
    if app.ai is not None and hasattr(app.ai, "close"):
        app.ai.close()
    pygame.quit()
