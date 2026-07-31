"""판정 로직 검증 — 9가지 조합 전수."""

import pytest

from rps.judge import KOREAN, Move, Outcome, judge, move_from_class_name

W, L, D = Outcome.WIN, Outcome.LOSE, Outcome.DRAW
R, P, S = Move.ROCK, Move.PAPER, Move.SCISSORS

# (사용자, AI, 기대 결과) — 3×3 전수
ALL_NINE = [
    (R, R, D), (R, P, L), (R, S, W),
    (P, R, W), (P, P, D), (P, S, L),
    (S, R, L), (S, P, W), (S, S, D),
]


@pytest.mark.parametrize("user, ai, expected", ALL_NINE)
def test_judge_all_nine_combinations(user, ai, expected):
    assert judge(user, ai) is expected


def test_covers_every_combination():
    """조합 누락 방지 — 9개가 모두 서로 달라야 한다."""
    assert len({(u, a) for u, a, _ in ALL_NINE}) == 9


def test_judge_is_antisymmetric():
    """무승부가 아니면 자리를 바꿨을 때 결과가 뒤집혀야 한다."""
    for user, ai, expected in ALL_NINE:
        if expected is D:
            assert judge(ai, user) is D
        else:
            flipped = judge(ai, user)
            assert flipped is (L if expected is W else W)


def test_draw_only_on_identical_moves():
    for user in Move:
        for ai in Move:
            assert (judge(user, ai) is D) == (user == ai)


@pytest.mark.parametrize("name, expected", [
    ("Rock", R), ("Paper", P), ("Scissors", S),
    ("rock", R), ("  SCISSORS  ", S),
])
def test_move_from_class_name(name, expected):
    """YOLO 클래스명은 대소문자·공백에 관계없이 매핑돼야 한다."""
    assert move_from_class_name(name) is expected


def test_move_from_unknown_class_name_is_none():
    assert move_from_class_name("Lizard") is None


def test_korean_labels_present():
    assert {KOREAN[m] for m in Move} == {"바위", "보", "가위"}
