"""가위바위보 승부 판정 + 라운드 상태 — 순수 로직, cv2/mediapipe import 금지.

승패는 결정적 규칙(가위>보>바위>가위)이므로 모델에 묻지 않는다. yolo/rps의
rps/judge.py + rps/game.py를 참고하되, 이 프로젝트는 손모양을 Gesture enum
(hand_lite.types)으로 이미 갖고 있으므로 Move enum을 새로 만들지 않고 그대로
재사용한다.
"""

from __future__ import annotations

import random
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum

from .types import Gesture

# 실제로 승부가 갈리는 손모양 3종 — UNKNOWN은 여기 끼워넣지 않는다(아래 judge 참고).
_PLAYABLE: tuple[Gesture, ...] = (Gesture.ROCK, Gesture.PAPER, Gesture.SCISSORS)

# key가 value를 이긴다 (가위바위보 표준 규칙)
_BEATS: dict[Gesture, Gesture] = {
    Gesture.ROCK: Gesture.SCISSORS,
    Gesture.PAPER: Gesture.ROCK,
    Gesture.SCISSORS: Gesture.PAPER,
}


class Result(Enum):
    """플레이어 관점의 라운드 결과."""

    WIN = "승리"
    LOSE = "패배"
    DRAW = "무승부"
    UNDECIDABLE = "판정불가"  # 플레이어 손이 UNKNOWN이라 승부 자체를 내지 않은 경우


def judge(player: Gesture, ai: Gesture) -> Result:
    """플레이어 관점의 승패를 반환한다.

    플레이어가 Gesture.UNKNOWN이면 억지로 승부를 내지 않고 UNDECIDABLE을
    반환한다 — gesture.classify()가 애매한 손모양을 UNKNOWN으로 남겨두는
    것과 같은 원칙: 확신 없는 입력을 억지로 셋 중 하나로 분류하지 않는다.
    ai는 AIMoveProvider.pick()이 _PLAYABLE 중에서만 고르므로 UNKNOWN이
    들어올 일이 없지만, 방어적으로 동일하게 처리한다.
    """
    if player is Gesture.UNKNOWN or ai is Gesture.UNKNOWN:
        return Result.UNDECIDABLE
    if player == ai:
        return Result.DRAW
    if _BEATS[player] == ai:
        return Result.WIN
    return Result.LOSE


class AIMoveProvider:
    """AI가 낼 패를 무작위로 고른다.

    random.Random(seed)를 주입할 수 있게 만들어, 테스트에서는 시드를 고정해
    "AI가 특정 시퀀스로 낸다"를 결정적으로 검증할 수 있다(실제 게임에서는
    seed=None으로 매 실행마다 다른 난수 스트림을 쓴다).
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def pick(self) -> Gesture:
        return self._rng.choice(_PLAYABLE)


class MoveStabilizer:
    """프레임 단위 흔들림 제거 — 최근 N프레임의 다수결로 손모양을 확정한다.

    왜 필요한가: MediaPipe 검출은 프레임마다 랜드마크가 미세하게 흔들려서
    (지터), 매 프레임 classify() 결과를 그대로 쓰면 손을 고정하고 있어도
    ROCK↔UNKNOWN을 오가며 판정이 요동친다. yolo/rps/rps/detector.py의
    MoveStabilizer와 동일한 아이디어(최근 window 프레임 중 최소 min_votes
    이상 나온 손모양만 "확정"으로 인정)를 그대로 가져온다 — 문제와 해법이
    동일해 새로 설계할 이유가 없다.
    """

    def __init__(self, window: int = 7, min_votes: int = 4) -> None:
        if min_votes > window:
            raise ValueError(f"min_votes({min_votes})는 window({window})보다 클 수 없습니다")
        self.window = window
        self.min_votes = min_votes
        self._buffer: deque[Gesture] = deque(maxlen=window)

    def update(self, gesture: Gesture) -> None:
        self._buffer.append(gesture)

    def stable_gesture(self) -> Gesture | None:
        """다수결로 min_votes 이상을 얻은 손모양이 있으면 그것을, 없으면 None.

        UNKNOWN도 표(vote) 하나로 집계한다 — "손이 안 잡히거나 애매한 상태가
        계속됐다"도 하나의 안정된 상태이기 때문이다. 다만 라운드 판정에서는
        UNKNOWN이 "확정"되어도 judge()가 UNDECIDABLE로 처리한다.
        """
        if not self._buffer:
            return None
        gesture, count = Counter(self._buffer).most_common(1)[0]
        return gesture if count >= self.min_votes else None

    def clear(self) -> None:
        self._buffer.clear()


@dataclass
class Score:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    def record(self, result: Result) -> None:
        if result is Result.WIN:
            self.wins += 1
        elif result is Result.LOSE:
            self.losses += 1
        elif result is Result.DRAW:
            self.draws += 1
        # UNDECIDABLE은 승부가 아니므로 점수에 반영하지 않는다.

    def as_tuple(self) -> tuple[int, int, int]:
        return self.wins, self.losses, self.draws


@dataclass
class RoundState:
    """라운드 1회의 결과 스냅샷 — app.py가 HUD에 그대로 넘길 수 있는 형태."""

    player: Gesture
    ai: Gesture
    result: Result


@dataclass
class Game:
    """라운드 진행 + 누적 점수. UI/카메라를 몰라도 되는 순수 상태 객체.

    타이밍(카운트다운 몇 초 등)은 이 프로젝트의 핵심 결함(미러링)과 무관하고
    app.py가 프레임 루프에서 직접 재생하기 쉬운 성질이라, 상태 머신을
    yolo/rps처럼 시간 기반으로 복잡하게 만들지 않는다. 대신 "안정화된
    손모양이 들어오면 즉시 판정" 이라는 최소 계약만 제공한다 — Rule 2
    (Simplicity First): 요청을 푸는 최소 코드만.
    """

    ai_provider: AIMoveProvider = field(default_factory=AIMoveProvider)
    score: Score = field(default_factory=Score)
    last_round: RoundState | None = None

    def play_round(self, stable_gesture: Gesture) -> RoundState:
        """안정화된 플레이어 손모양 1개로 라운드 1회를 즉시 판정한다.

        stable_gesture가 UNKNOWN이면 AI도 패를 내지 않고(굳이 낼 이유가
        없다) UNDECIDABLE 라운드로 기록한다 — 점수에는 반영되지 않는다.
        """
        if stable_gesture is Gesture.UNKNOWN:
            round_state = RoundState(player=Gesture.UNKNOWN, ai=Gesture.UNKNOWN, result=Result.UNDECIDABLE)
        else:
            ai_gesture = self.ai_provider.pick()
            result = judge(stable_gesture, ai_gesture)
            round_state = RoundState(player=stable_gesture, ai=ai_gesture, result=result)
            self.score.record(result)

        self.last_round = round_state
        return round_state

    def reset_score(self) -> None:
        self.score = Score()
