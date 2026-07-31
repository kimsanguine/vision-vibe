"""손가락 개수 → 제스처 매핑 규칙 검증.

여기서 고정하는 '의도(WHY)'는 두 가지다.
  1. 0/2/5 개만 확신하고, 나머지는 **억지로 판정하지 않는다**.
  2. 골 0개는 주먹과 손가락 1개 사이에서 모호하므로, solidity 로만 가른다.
"""

from __future__ import annotations

import pytest

from rps_classic.fingers import FIST_SOLIDITY, count_extended_fingers
from rps_classic.gesture import classify
from rps_classic.types import Gesture


# ── 손가락 개수 → 제스처 ────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("fingers", "expected"),
    [
        (0, Gesture.ROCK),
        (2, Gesture.SCISSORS),
        (5, Gesture.PAPER),
    ],
)
def test_확신할_수_있는_개수만_손모양으로_매핑된다(fingers, expected):
    assert classify(fingers) is expected


@pytest.mark.parametrize("fingers", [1, 3, 4, 6, 7, 10])
def test_애매한_개수는_UNKNOWN이다(fingers):
    """1·3·4개를 '가장 가까운 것'으로 반올림하지 않는다.

    반올림하면 게임은 언제나 답을 내놓지만 그 답이 맞을 근거가 없다.
    mediapipe/hand_lite / yolo/rps 와 기권 기준을 맞춰야 3자 정확도 비교가
    같은 잣대 위에 놓인다 — 한 구현만 억지 판정을 하면 그쪽 정확도가 실제보다
    좋아 보인다.
    """
    assert classify(fingers) is Gesture.UNKNOWN


@pytest.mark.parametrize("fingers", [-1, -5])
def test_음수_입력도_UNKNOWN으로_떨어진다(fingers):
    """방어적 계약 — 상류 버그가 조용히 ROCK 으로 둔갑하지 않아야 한다."""
    assert classify(fingers) is Gesture.UNKNOWN


# ── 골 개수 + solidity → 손가락 개수 ────────────────────────────────────

@pytest.mark.parametrize(
    ("gaps", "expected"),
    [(1, 2), (2, 3), (3, 4), (4, 5)],
)
def test_골이_있으면_손가락은_골더하기1이다(gaps, expected):
    """골이 1개 이상이면 solidity 와 무관하게 gaps+1 이다."""
    assert count_extended_fingers(gaps, solidity=0.5) == expected
    assert count_extended_fingers(gaps, solidity=0.99) == expected


def test_골이_0개면_solidity가_주먹과_한손가락을_가른다():
    """0골 구간의 모호성 해소 규칙을 고정한다.

    이 분기가 없으면 gaps+1 규칙 때문에 주먹이 항상 손가락 1개로 읽혀
    ROCK 이 **영원히 검출 불가능**해진다. 그래서 이건 편의 기능이 아니라
    ROCK 검출의 필요조건이다.
    """
    assert count_extended_fingers(0, solidity=0.95) == 0   # 꽉 찬 실루엣 = 주먹
    assert count_extended_fingers(0, solidity=0.50) == 1   # 뾰족하게 튀어나옴 = 손가락 1개


def test_solidity_임계값_경계에서_주먹쪽으로_붙는다():
    """경계값 정확히 = 임계값이면 주먹으로 본다(>= 비교)."""
    assert count_extended_fingers(0, solidity=FIST_SOLIDITY) == 0
    assert count_extended_fingers(0, solidity=FIST_SOLIDITY - 1e-9) == 1


def test_주먹은_ROCK_손가락하나는_UNKNOWN으로_이어진다():
    """count_extended_fingers → classify 연결이 의도대로 흐르는지 확인."""
    assert classify(count_extended_fingers(0, solidity=0.90)) is Gesture.ROCK
    assert classify(count_extended_fingers(0, solidity=0.60)) is Gesture.UNKNOWN
    assert classify(count_extended_fingers(1, solidity=0.70)) is Gesture.SCISSORS
    assert classify(count_extended_fingers(4, solidity=0.70)) is Gesture.PAPER
