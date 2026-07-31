"""라운드 진행 상태 머신.

READY → COUNTDOWN(3·2·1) → RESULT(결과 표시) → READY
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto

from .ai_move import AIMoveProvider
from .judge import Move, Outcome, judge

COUNTDOWN_SECONDS = 3.0
RESULT_SECONDS = 2.5


class Phase(Enum):
    READY = auto()
    COUNTDOWN = auto()
    RESULT = auto()


@dataclass
class Score:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    def record(self, outcome: Outcome) -> None:
        if outcome is Outcome.WIN:
            self.wins += 1
        elif outcome is Outcome.LOSE:
            self.losses += 1
        else:
            self.draws += 1

    def as_tuple(self) -> tuple[int, int, int]:
        return self.wins, self.losses, self.draws


@dataclass
class Game:
    ai_provider: AIMoveProvider
    phase: Phase = Phase.READY
    score: Score = field(default_factory=Score)
    user_move: Move | None = None
    ai_move: Move | None = None
    outcome: Outcome | None = None
    missed: bool = False  # 카운트다운 종료 시 손모양을 못 잡은 경우
    _phase_started: float = field(default_factory=time.monotonic)

    # --- 전이 ---

    def start_round(self) -> None:
        if self.phase is not Phase.READY:
            return
        self.user_move = None
        self.ai_move = None
        self.outcome = None
        self.missed = False
        self._enter(Phase.COUNTDOWN)

    def reset_score(self) -> None:
        self.score = Score()

    def _enter(self, phase: Phase) -> None:
        self.phase = phase
        self._phase_started = time.monotonic()

    def _elapsed(self) -> float:
        return time.monotonic() - self._phase_started

    # --- 매 프레임 호출 ---

    def update(self, stable_move: Move | None) -> None:
        """현재 프레임의 확정 손모양을 받아 상태를 진행한다."""
        if self.phase is Phase.COUNTDOWN and self._elapsed() >= COUNTDOWN_SECONDS:
            self._resolve(stable_move)
        elif self.phase is Phase.RESULT and self._elapsed() >= RESULT_SECONDS:
            self._enter(Phase.READY)

    def _resolve(self, stable_move: Move | None) -> None:
        if stable_move is None:
            self.missed = True
            self.user_move = None
            self.ai_move = None
            self.outcome = None
        else:
            self.missed = False
            self.user_move = stable_move
            self.ai_move = self.ai_provider.pick()
            self.outcome = judge(self.user_move, self.ai_move)
            self.score.record(self.outcome)
        self._enter(Phase.RESULT)

    # --- 표시용 ---

    def countdown_number(self) -> int | None:
        if self.phase is not Phase.COUNTDOWN:
            return None
        remaining = COUNTDOWN_SECONDS - self._elapsed()
        return max(1, min(int(COUNTDOWN_SECONDS), int(remaining) + 1))
