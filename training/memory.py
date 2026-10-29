"""學習資料：位子牌組記憶（PositionCardMemory）與對手落子記憶（OpponentMemory）。

位子牌組以「稀疏」方式儲存：只記錄牌數與初始值不同的位置，
沒有記錄的合法位置一律視為 initial_weight 張牌。
這與「每個局面都放滿初始牌」在數學上完全等價，但能大幅降低檔案大小。
"""

from __future__ import annotations

from core.board import SIZE


def action_to_str(a: int) -> str:
    r, c = divmod(a, SIZE)
    return f"{r},{c}"


def str_to_action(s: str) -> int:
    r, c = s.split(",")
    return int(r) * SIZE + int(c)


class PositionCardMemory:
    def __init__(self, initial_weight: int = 10, min_weight: int = 1):
        if initial_weight <= 0:
            raise ValueError("initial_weight 必須 > 0")
        if min_weight < 0:
            raise ValueError("min_weight 必須 >= 0")
        self.initial_weight = initial_weight
        self.min_weight = min_weight
        self.states: dict[str, dict[int, int]] = {}  # 局面 -> {位置: 牌數}（僅記錄非初始值）
        self.visits: dict[str, int] = {}             # 局面 -> 被 AI 選擇過的次數
        self.deck_resets = 0                          # 全部牌數歸零而重設的次數

    def __len__(self) -> int:
        return len(self.states)

    def has_state(self, key: str) -> bool:
        return key in self.states

    def weight(self, key: str, action: int) -> int:
        return self.states.get(key, {}).get(action, self.initial_weight)

    def deck(self, key: str, legal: list[int]) -> list[int]:
        """legal 中每個位置的牌數（順序與 legal 相同）。"""
        d = self.states.get(key)
        if d is None:
            return [self.initial_weight] * len(legal)
        w0 = self.initial_weight
        return [d.get(a, w0) for a in legal]

    def register(self, key: str) -> None:
        """記錄 AI 造訪了此局面（學習模式才呼叫）。"""
        self.states.setdefault(key, {})
        self.visits[key] = self.visits.get(key, 0) + 1

    def reset_deck(self, key: str) -> None:
        """所有合法位置牌數皆為 0 時，把該局面恢復成初始牌數。"""
        self.states[key] = {}
        self.deck_resets += 1

    def adjust(self, key: str, action: int, delta: int) -> tuple[int, int]:
        """增減牌數並套用下限，回傳 (調整前, 調整後)。"""
        d = self.states.setdefault(key, {})
        before = d.get(action, self.initial_weight)
        after = max(self.min_weight, before + delta)
        if after == self.initial_weight:
            d.pop(action, None)
        else:
            d[action] = after
        return before, after

    # ---- 統計 / 保存 ------------------------------------------------------
    def stats(self) -> dict:
        changed = sum(len(d) for d in self.states.values())
        return {
            "learned_states": len(self.states),
            "changed_entries": changed,
            "deck_resets": self.deck_resets,
            "total_visits": sum(self.visits.values()),
        }

    def to_dict(self) -> dict:
        return {
            "initial_weight": self.initial_weight,
            "min_weight": self.min_weight,
            "deck_resets": self.deck_resets,
            "states": {k: {action_to_str(a): w for a, w in sorted(d.items())} for k, d in self.states.items()},
            "visits": self.visits,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PositionCardMemory":
        m = cls(data["initial_weight"], data["min_weight"])
        m.deck_resets = data.get("deck_resets", 0)
        m.states = {k: {str_to_action(a): int(w) for a, w in d.items()} for k, d in data["states"].items()}
        m.visits = {k: int(v) for k, v in data.get("visits", {}).items()}
        return m


class OpponentMemory:
    """記錄對手在每個局面（以 AI 視角編碼）實際下的位置與次數。

    states[key][action] = [次數, 這一手直接讓對手五連獲勝的次數]
    """

    def __init__(self):
        self.states: dict[str, dict[int, list[int]]] = {}

    def __len__(self) -> int:
        return len(self.states)

    def record(self, key: str, action: int, won: bool) -> None:
        entry = self.states.setdefault(key, {}).setdefault(action, [0, 0])
        entry[0] += 1
        if won:
            entry[1] += 1

    def replies(self, key: str) -> dict[int, list[int]]:
        return self.states.get(key, {})

    def predicted(self, key: str) -> int | None:
        """對手在此局面最常下的位置（無紀錄則 None）。"""
        d = self.states.get(key)
        if not d:
            return None
        return max(d, key=lambda a: (d[a][0], -a))

    def has_winning_reply(self, key: str) -> bool:
        return any(v[1] > 0 for v in self.states.get(key, {}).values())

    def stats(self) -> dict:
        obs = sum(v[0] for d in self.states.values() for v in d.values())
        repeated = [d for d in self.states.values() if sum(v[0] for v in d.values()) >= 2]
        consistent = sum(1 for d in repeated if len(d) == 1)
        winning = sum(1 for d in self.states.values() if any(v[1] for v in d.values()))
        return {
            "opponent_states": len(self.states),
            "opponent_observations": obs,
            "states_seen_twice_or_more": len(repeated),
            "consistent_reply_ratio": round(consistent / len(repeated), 4) if repeated else None,
            "states_with_opponent_winning_reply": winning,
        }

    def top_states(self, n: int = 5) -> list[dict]:
        """出現最多次的對手局面與其落子分佈（供報告使用）。"""
        items = sorted(self.states.items(), key=lambda kv: -sum(v[0] for v in kv[1].values()))[:n]
        return [
            {"state": k, "replies": {action_to_str(a): {"count": v[0], "wins": v[1]} for a, v in sorted(d.items(), key=lambda x: -x[1][0])}}
            for k, d in items
        ]

    def to_dict(self) -> dict:
        return {k: {action_to_str(a): v for a, v in d.items()} for k, d in self.states.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "OpponentMemory":
        m = cls()
        m.states = {k: {str_to_action(a): [int(v[0]), int(v[1])] for a, v in d.items()} for k, d in data.items()}
        return m


class PatternCardMemory:
    """棋形牌組：局部棋形 -> 牌數（修正後 AI 使用）。未出現過的棋形視為 initial_weight。"""

    def __init__(self, initial_weight: int = 10, min_weight: int = 1):
        if initial_weight <= 0:
            raise ValueError("initial_weight 必須 > 0")
        if min_weight < 0:
            raise ValueError("min_weight 必須 >= 0")
        self.initial_weight = initial_weight
        self.min_weight = min_weight
        self.cards: dict[str, int] = {}
        self.uses: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.cards)

    def get(self, key: str) -> int:
        return self.cards.get(key, self.initial_weight)

    def adjust(self, key: str, delta: int) -> tuple[int, int]:
        before = self.cards.get(key, self.initial_weight)
        after = max(self.min_weight, before + delta)
        self.cards[key] = after
        self.uses[key] = self.uses.get(key, 0) + 1
        return before, after

    def stats(self) -> dict:
        return {"learned_patterns": len(self.cards), "total_updates": sum(self.uses.values())}

    def to_dict(self) -> dict:
        return {"initial_weight": self.initial_weight, "min_weight": self.min_weight,
                "cards": self.cards, "uses": self.uses}

    @classmethod
    def from_dict(cls, data: dict) -> "PatternCardMemory":
        m = cls(data["initial_weight"], data["min_weight"])
        m.cards = {k: int(v) for k, v in data["cards"].items()}
        m.uses = {k: int(v) for k, v in data.get("uses", {}).items()}
        return m


class OpponentPatternMemory:
    """以棋形記憶對手的落子偏好。

    seen[key]   = 對手輪到時，盤面上「具有此棋形的空位」出現的次數
    chosen[key] = 對手實際選擇了具有此棋形的位置的次數
    棋形以「對手自己的視角」計算，因此 chosen/seen 代表對手偏好這種位置的程度。
    """

    def __init__(self):
        self.seen: dict[str, int] = {}
        self.chosen: dict[str, int] = {}
        self.observations = 0

    def __len__(self) -> int:
        return len(self.chosen)

    def record(self, available_keys: list[list[str]], chosen_index: int) -> None:
        self.observations += 1
        for i, keys in enumerate(available_keys):
            for k in keys:
                self.seen[k] = self.seen.get(k, 0) + 1
                if i == chosen_index:
                    self.chosen[k] = self.chosen.get(k, 0) + 1

    def preference(self, keys: list[str]) -> float:
        """0~1：對手選擇這種位置的經驗比率（取四個方向中最高者）。"""
        best = 0.0
        for k in keys:
            s = self.seen.get(k, 0)
            if s:
                best = max(best, self.chosen.get(k, 0) / (s + 1))
        return best

    def stats(self) -> dict:
        return {"opponent_observations": self.observations, "opponent_patterns": len(self.seen),
                "opponent_chosen_patterns": len(self.chosen)}

    def top_patterns(self, n: int = 10, min_seen: int = 20) -> list[dict]:
        rows = [(k, self.chosen.get(k, 0), s) for k, s in self.seen.items() if s >= min_seen]
        rows.sort(key=lambda x: -(x[1] / (x[2] + 1)))
        return [{"pattern": k, "chosen": c, "seen": s, "rate": round(c / (s + 1), 3)} for k, c, s in rows[:n]]

    def to_dict(self) -> dict:
        return {"seen": self.seen, "chosen": self.chosen, "observations": self.observations}

    @classmethod
    def from_dict(cls, data: dict) -> "OpponentPatternMemory":
        m = cls()
        m.seen = {k: int(v) for k, v in data.get("seen", {}).items()}
        m.chosen = {k: int(v) for k, v in data.get("chosen", {}).items()}
        m.observations = int(data.get("observations", 0))
        return m
