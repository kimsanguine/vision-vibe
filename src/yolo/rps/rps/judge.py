"""가위바위보 판정 — 순수 코드 로직 (LLM 미사용).

승패는 결정적 규칙이므로 모델에 묻지 않는다.
"""

from __future__ import annotations

from enum import Enum


class Move(str, Enum):
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"


class Outcome(str, Enum):
    WIN = "win"    # 사용자 승
    LOSE = "lose"  # 사용자 패
    DRAW = "draw"


# key가 value를 이긴다
_BEATS: dict[Move, Move] = {
    Move.ROCK: Move.SCISSORS,
    Move.PAPER: Move.ROCK,
    Move.SCISSORS: Move.PAPER,
}

KOREAN: dict[Move, str] = {
    Move.ROCK: "바위",
    Move.PAPER: "보",
    Move.SCISSORS: "가위",
}

EMOJI_FALLBACK: dict[Move, str] = {
    Move.ROCK: "✊",
    Move.PAPER: "✋",
    Move.SCISSORS: "✌",
}

OUTCOME_KOREAN: dict[Outcome, str] = {
    Outcome.WIN: "승리!",
    Outcome.LOSE: "패배",
    Outcome.DRAW: "무승부",
}

# YOLO 모델 클래스명(Paper/Rock/Scissors) → Move.
# 클래스 인덱스는 모델마다 다를 수 있으므로 반드시 '이름' 기준으로 매핑한다.
CLASS_NAME_TO_MOVE: dict[str, Move] = {
    "rock": Move.ROCK,
    "paper": Move.PAPER,
    "scissors": Move.SCISSORS,
}


def move_from_class_name(name: str) -> Move | None:
    """YOLO 클래스명을 Move로 변환. 알 수 없는 이름이면 None."""
    return CLASS_NAME_TO_MOVE.get(name.strip().lower())


def judge(user: Move, ai: Move) -> Outcome:
    """사용자 관점의 승패를 반환한다."""
    if user == ai:
        return Outcome.DRAW
    if _BEATS[user] == ai:
        return Outcome.WIN
    return Outcome.LOSE
