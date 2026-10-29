"""外部引擎 Rapfi（https://github.com/dhbloo/rapfi）的代理。

透過 Gomocup / Piskvork 協定（標準輸入輸出）與引擎溝通：
    START 7                 7×7 棋盤
    INFO rule 0             自由五子棋（五連以上即勝），與本專案規則相同
    INFO thread_num 1       單執行緒，搭配 max_node 讓結果可重現
    INFO max_node N         限制每手搜尋節點數，用來調整強度
    BOARD ... DONE          每一手都送出完整盤面（1 = 引擎自己、2 = 對手），引擎回傳 "x,y"
座標：協定的 x 是行（column）、y 是列（row）。

Rapfi 只作為「強力 AI」與評估對手，不參與位子牌學習。
找不到執行檔時丟出 EngineUnavailableError，呼叫端可以略過相關實驗並註明原因。
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import threading

from core.board import BLACK, SIZE, Board

from .base_agent import Agent

_COORD = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*$")
DEFAULT_PATHS = ("engines/rapfi/pbrain-rapfi", "engines/rapfi/pbrain-rapfi.exe")


class EngineUnavailableError(RuntimeError):
    pass


def find_rapfi(path: str | None = None) -> str:
    """依序嘗試：參數 → 環境變數 RAPFI_PATH → engines/rapfi/ → PATH。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [path, os.environ.get("RAPFI_PATH")]
    candidates += [os.path.join(root, p) for p in DEFAULT_PATHS]
    for c in candidates:
        if not c:
            continue
        for p in (c, c + ".exe"):
            full = p if os.path.isabs(p) else os.path.join(root, p)
            if os.path.isfile(full) and os.access(full, os.X_OK):
                return full
    on_path = shutil.which("pbrain-rapfi")
    if on_path:
        return on_path
    raise EngineUnavailableError(
        "找不到 Rapfi 執行檔。請把 pbrain-rapfi(.exe) 放到 engines/rapfi/，"
        "或設定環境變數 RAPFI_PATH，或在設定檔中指定 path。")


class RapfiAgent(Agent):
    kind = "rapfi"
    version = "external"

    def __init__(self, path: str | None = None, max_node: int | None = 20000, max_depth: int | None = None,
                 timeout_turn_ms: int = 5000, name: str | None = None, seed: int | None = None):
        super().__init__(name or "rapfi", seed)
        self.path = find_rapfi(path)
        self.max_node = max_node
        self.max_depth = max_depth
        self.timeout_turn_ms = timeout_turn_ms
        self.proc: subprocess.Popen | None = None
        self.about = ""

    def params(self) -> dict:
        return {"max_node": self.max_node, "max_depth": self.max_depth,
                "timeout_turn_ms": self.timeout_turn_ms, "engine": self.about or "rapfi"}

    # ---- 程序管理 ---------------------------------------------------------
    def _send(self, line: str) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _pump(self, stream, q: "queue.Queue[str | None]") -> None:
        for line in stream:
            q.put(line.strip())
        q.put(None)

    def _next_line(self, timeout: float | None) -> str:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError from None
        if line is None:
            raise EngineUnavailableError("Rapfi 程序已結束")
        return line

    def _read_until(self, pred, what: str, timeout: float | None = 120.0) -> str:
        while True:
            try:
                line = self._next_line(timeout)
            except TimeoutError:
                raise RuntimeError(f"等待 Rapfi {what} 逾時") from None
            if line.startswith("ERROR"):
                raise RuntimeError(f"Rapfi 回報錯誤：{line}")
            if pred(line):
                return line

    def _sync(self) -> None:
        """確認引擎已閒置。

        Rapfi 輸出落子後，「思考中」狀態會稍晚才結束，這段時間收到的指令會被丟棄。
        因此送出新盤面前先送 ABOUT，直到收到回應（代表已閒置）為止。
        """
        for _ in range(200):
            self._send("ABOUT")
            try:
                while True:
                    line = self._next_line(0.25)
                    if "name=" in line:
                        return
            except TimeoutError:
                continue
        raise RuntimeError("Rapfi 長時間沒有回應")

    def _ensure_started(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        self.proc = subprocess.Popen([self.path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                     cwd=os.path.dirname(self.path))
        self.lines: "queue.Queue[str | None]" = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc.stdout, self.lines), daemon=True).start()
        self._send("ABOUT")
        self.about = self._read_until(lambda l: l.startswith("name") or "name=" in l, "ABOUT")
        self._send(f"START {SIZE}")
        self._read_until(lambda l: l == "OK", "START 回應")
        self._send("INFO rule 0")
        self._send("INFO thread_num 1")
        self._send("INFO pondering 0")
        self._send(f"INFO timeout_turn {self.timeout_turn_ms}")
        self._send("INFO timeout_match 0")
        self._send(f"INFO time_left {2**31 - 1}")
        if self.max_node:
            self._send(f"INFO max_node {self.max_node}")
        if self.max_depth:
            self._send(f"INFO max_depth {self.max_depth}")

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self._send("END")
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
        self.proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ---- 對弈 -------------------------------------------------------------
    def start_game(self, color: int, seed: int | None = None) -> None:
        super().start_game(color, seed)
        self._ensure_started()
        self._sync()
        self._send("YXHASHCLEAR")  # 清除前一局的置換表，讓每局結果只取決於盤面
        self.history: list[tuple[int, int]] = []  # (index, color)

    def observe_move(self, board_before: Board, move: int, color: int) -> None:
        self.history.append((move, color))

    def select_move(self, board: Board, color: int) -> int:
        self._ensure_started()
        # 依實際落子順序送出（BOARD 需要正確順序才能還原局面）
        seq = self.history
        if len(seq) != board.stone_count():
            raise RuntimeError("Rapfi 代理的落子歷史與棋盤不一致")
        self._sync()
        self._send("BOARD")
        for idx, c in seq:
            r, col = divmod(idx, SIZE)
            self._send(f"{col},{r},{1 if c == color else 2}")
        self._send("DONE")
        line = self._read_until(lambda l: _COORD.match(l) is not None, "落子")
        m = _COORD.match(line)
        x, y = int(m.group(1)), int(m.group(2))
        idx = y * SIZE + x
        if not (0 <= x < SIZE and 0 <= y < SIZE) or board.cells[idx] != 0:
            raise RuntimeError(f"Rapfi 回傳非法位置 {line}")
        return idx


def rapfi_available(path: str | None = None) -> bool:
    try:
        find_rapfi(path)
        return True
    except EngineUnavailableError:
        return False


__all__ = ["RapfiAgent", "EngineUnavailableError", "find_rapfi", "rapfi_available", "BLACK"]
