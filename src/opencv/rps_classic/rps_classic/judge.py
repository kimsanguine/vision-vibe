"""5단계 — 승패 판정. 순수 로직, cv2 import 금지.

승패는 결정적 규칙(가위>보>바위>가위)이라 모델에 물을 이유가 없다.
yolo/rps/rps/judge.py 와 mediapipe/hand_lite/hand_lite/rps.py 가 이미 같은
규칙을 갖고 있고, 문제와 해법이 동일하므로 새로 설계하지 않고 계약만 맞춘다.

한 가지만 다르다: yolo/rps 의 judge 는 Outcome 3종(win/lose/draw)뿐이라
UNKNOWN 입력을 표현할 수 없다. 이 프로젝트는 mediapipe/hand_lite 쪽 계약
(UNDECIDABLE 추가)을 따른다 — 전통 기법은 UNKNOWN 이 훨씬 자주 나오므로
"판정불가"를 무승부로 뭉개면 성능이 실제보다 좋아 보인다.
"""

from __future__ import annotations

import random

from .types import Gesture, Result

# 실제로 승부가 갈리는 손모양 3종. UNKNOWN 은 여기 넣지 않는다.
_PLAYABLE: tuple[Gesture, ...] = (Gesture.ROCK, Gesture.PAPER, Gesture.SCISSORS)

# key 가 value 를 이긴다
_BEATS: dict[Gesture, Gesture] = {
    Gesture.ROCK: Gesture.SCISSORS,
    Gesture.PAPER: Gesture.ROCK,
    Gesture.SCISSORS: Gesture.PAPER,
}


def judge(player: Gesture, ai: Gesture) -> Result:
    """플레이어 관점의 승패를 반환한다.

    어느 한쪽이라도 UNKNOWN 이면 무승부가 아니라 UNDECIDABLE 이다.
    무승부로 처리하면 "인식 실패"가 "비겼다"라는 정상 결과로 둔갑해,
    파이프라인이 얼마나 자주 실패하는지가 통계에서 사라진다.
    """
    if player is Gesture.UNKNOWN or ai is Gesture.UNKNOWN:
        return Result.UNDECIDABLE
    if player is ai:
        return Result.DRAW
    if _BEATS[player] is ai:
        return Result.WIN
    return Result.LOSE


class AIMoveProvider:
    """AI 가 낼 패를 무작위로 고른다.

    seed 를 주입할 수 있게 해서 테스트에서 AI 수를 결정적으로 고정한다.
    실제 게임에서는 seed=None 으로 매 실행마다 다른 난수 스트림을 쓴다.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def pick(self) -> Gesture:
        return self._rng.choice(_PLAYABLE)
