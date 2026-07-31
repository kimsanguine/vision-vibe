"""승패 판정 검증 — 3x3 = 9가지 조합 전수 + UNKNOWN 처리.

승패 규칙은 경우의 수가 9개뿐인 유한 함수라 **전수 검증**이 가능하다.
샘플링할 이유가 없으므로 9칸을 표로 적어 전부 확인한다.
"""

from __future__ import annotations

import pytest

from rps_classic.judge import AIMoveProvider, judge
from rps_classic.types import Gesture, Result

R, P, S = Gesture.ROCK, Gesture.PAPER, Gesture.SCISSORS
WIN, LOSE, DRAW = Result.WIN, Result.LOSE, Result.DRAW


@pytest.mark.parametrize(
    ("player", "ai", "expected"),
    [
        # 무승부 3칸
        (R, R, DRAW),
        (P, P, DRAW),
        (S, S, DRAW),
        # 플레이어 승 3칸 (바위>가위, 보>바위, 가위>보)
        (R, S, WIN),
        (P, R, WIN),
        (S, P, WIN),
        # 플레이어 패 3칸
        (R, P, LOSE),
        (P, S, LOSE),
        (S, R, LOSE),
    ],
)
def test_9가지_조합_전수(player, ai, expected):
    assert judge(player, ai) is expected


@pytest.mark.parametrize(
    ("player", "ai"),
    [
        (Gesture.UNKNOWN, R),
        (Gesture.UNKNOWN, P),
        (Gesture.UNKNOWN, S),
        (R, Gesture.UNKNOWN),
        (P, Gesture.UNKNOWN),
        (S, Gesture.UNKNOWN),
        (Gesture.UNKNOWN, Gesture.UNKNOWN),
    ],
)
def test_UNKNOWN이_끼면_무승부가_아니라_판정불가다(player, ai):
    """가장 중요한 계약.

    UNKNOWN 을 DRAW 로 뭉개면 "인식 실패"가 "비겼다"라는 정상 결과로 둔갑해,
    파이프라인이 얼마나 자주 실패하는지가 점수 통계에서 사라진다. 이 프로젝트는
    전통 기법이라 UNKNOWN 이 매우 자주 나오므로(실측 4/6), 이 구분이 없으면
    3자 비교 자체가 왜곡된다.
    """
    assert judge(player, ai) is Result.UNDECIDABLE


def test_UNKNOWN_UNKNOWN은_같은_값이지만_무승부가_아니다():
    """player == ai 검사보다 UNKNOWN 검사가 **먼저** 와야 한다는 순서 계약.

    순서를 뒤집으면 (UNKNOWN, UNKNOWN) 이 DRAW 로 빠져나간다 — 조용한 버그다.
    """
    assert judge(Gesture.UNKNOWN, Gesture.UNKNOWN) is not Result.DRAW
    assert judge(Gesture.UNKNOWN, Gesture.UNKNOWN) is Result.UNDECIDABLE


def test_승패는_비대칭이다():
    """judge(a,b) 가 WIN 이면 judge(b,a) 는 반드시 LOSE 여야 한다."""
    for a in (R, P, S):
        for b in (R, P, S):
            forward, backward = judge(a, b), judge(b, a)
            if forward is WIN:
                assert backward is LOSE
            elif forward is LOSE:
                assert backward is WIN
            else:
                assert backward is DRAW


# ── AI 수 선택 ──────────────────────────────────────────────────────────

def test_AI는_UNKNOWN을_내지_않는다():
    """AI 가 UNKNOWN 을 내면 모든 라운드가 판정불가가 된다."""
    provider = AIMoveProvider(seed=20260729)
    picks = {provider.pick() for _ in range(300)}
    assert Gesture.UNKNOWN not in picks
    assert picks == {R, P, S}, "300회 안에 세 손모양이 모두 나와야 한다"


def _sequence(seed: int, n: int) -> list[Gesture]:
    """provider **하나**에서 n번 뽑는다.

    매 호출마다 새 provider 를 만들면 난수 스트림이 매번 처음으로 되감겨
    같은 값만 n개 나온다 — 수열을 검증한다고 믿으면서 실제로는 첫 값 하나만
    반복 확인하게 된다.
    """
    provider = AIMoveProvider(seed=seed)
    return [provider.pick() for _ in range(n)]


def test_같은_시드는_같은_수열을_낸다():
    """테스트에서 AI 수를 고정할 수 있어야 게임 로직을 결정적으로 검증한다."""
    assert _sequence(7, 20) == _sequence(7, 20)


def test_다른_시드는_다른_수열을_낸다():
    assert _sequence(1, 30) != _sequence(2, 30)


def test_한_provider는_매번_같은_값만_내지_않는다():
    """seed 를 고정해도 스트림은 진행되어야 한다 — 상태가 없는 구현 방지."""
    assert len(set(_sequence(20260729, 30))) > 1
