"""棋局紀錄：每局一行 JSON（JSONL），可重播驗證。"""

from __future__ import annotations

import csv
import gzip
import json
import os
from typing import Iterable, Iterator


class MatchLogger:
    """把對局紀錄逐行寫入 JSONL。預設拒絕覆蓋既有檔案，以保留原始實驗數據。"""

    def __init__(self, path: str, overwrite: bool = False):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"{path} 已存在；為保留原始紀錄不會覆蓋（可改用新的實驗名稱或 --force）")
        self.path = path
        self._f = gzip.open(path, "wt", encoding="utf-8") if path.endswith(".gz") else open(path, "w", encoding="utf-8")
        self.count = 0

    def write(self, record: dict) -> None:
        self._f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.count += 1

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "MatchLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_matches(path: str) -> Iterator[dict]:
    opener = gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")
    with opener as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_match(path: str, game_id: str) -> dict | None:
    for rec in read_matches(path):
        if rec.get("game_id") == game_id:
            return rec
    return None


def write_csv(path: str, rows: Iterable[dict], overwrite: bool = False) -> None:
    rows = list(rows)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} 已存在；為保留原始紀錄不會覆蓋")
    if not rows:
        open(path, "w", encoding="utf-8").close()
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
