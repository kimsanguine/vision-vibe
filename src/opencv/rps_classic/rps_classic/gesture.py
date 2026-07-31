"""4단계 — 손가락 개수 → 가위/바위/보 규칙 매핑.

순수 if 문이다. LLM 을 쓰지 않는다: 입력이 정수 하나이고 출력이 4가지뿐인
전결정 함수라 모델이 기여할 여지가 없다(전역 규칙 Rule 5 — 결정적 로직에
모델을 쓰지 않는다).

핵심 원칙 — UNKNOWN 을 억지로 판정하지 않는다
──────────────────────────────────────────────
1·3·4개는 셋 중 어느 것으로도 확신할 수 없는 값이다. "가장 가까운 것"으로
반올림하면 게임은 항상 뭔가를 내놓지만, 그 답이 맞을 이유가 없다.
mediapipe/hand_lite/hand_lite/rps.py 가 같은 원칙을 명시하고 있어 그대로
따른다 — 3자 비교에서 세 구현의 '기권 기준'이 다르면 정확도 비교가
사과와 오렌지가 된다.
"""

from __future__ import annotations

from .types import Gesture

# 편 손가락 개수 → 손모양. 여기 없는 개수는 전부 UNKNOWN.
_FINGERS_TO_GESTURE: dict[int, Gesture] = {
    0: Gesture.ROCK,      # 주먹
    2: Gesture.SCISSORS,  # 검지 + 중지
    5: Gesture.PAPER,     # 다섯 손가락
}


def classify(fingers: int) -> Gesture:
    """편 손가락 개수를 손모양으로 변환한다. 애매하면 UNKNOWN."""
    return _FINGERS_TO_GESTURE.get(fingers, Gesture.UNKNOWN)
