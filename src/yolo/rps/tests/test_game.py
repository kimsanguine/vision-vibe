"""라운드 상태 머신 검증."""

import time

import pytest

from rps.ai_move import AIMoveProvider
from rps.game import COUNTDOWN_SECONDS, Game, Phase
from rps.judge import Move, Outcome


class FixedAI(AIMoveProvider):
    """AI 패를 고정해 판정 결과를 결정적으로 만든다."""

    def __init__(self, move: Move):
        super().__init__()
        self._move = move

    def pick(self) -> Move:
        return self._move


def _game(ai_move: Move = Move.SCISSORS) -> Game:
    return Game(ai_provider=FixedAI(ai_move))


def test_starts_in_ready():
    assert _game().phase is Phase.READY


def test_space_starts_countdown():
    g = _game()
    g.start_round()
    assert g.phase is Phase.COUNTDOWN
    assert g.countdown_number() == 3


def test_start_round_ignored_while_countdown():
    g = _game()
    g.start_round()
    started = g._phase_started
    g.start_round()
    assert g._phase_started == started


def test_resolve_records_win():
    g = _game(Move.SCISSORS)
    g.start_round()
    g._phase_started = time.monotonic() - COUNTDOWN_SECONDS  # 카운트다운 종료 시점으로 이동
    g.update(Move.ROCK)
    assert g.phase is Phase.RESULT
    assert g.outcome is Outcome.WIN
    assert g.score.as_tuple() == (1, 0, 0)


def test_resolve_records_loss_and_draw():
    g = _game(Move.PAPER)
    for expected_score in [(0, 1, 0), (0, 2, 0)]:
        g.phase = Phase.READY
        g.start_round()
        g._phase_started = time.monotonic() - COUNTDOWN_SECONDS
        g.update(Move.ROCK)
        assert g.score.as_tuple() == expected_score


def test_missed_detection_does_not_change_score():
    g = _game()
    g.start_round()
    g._phase_started = time.monotonic() - COUNTDOWN_SECONDS
    g.update(None)
    assert g.phase is Phase.RESULT
    assert g.missed is True
    assert g.outcome is None
    assert g.score.as_tuple() == (0, 0, 0)


def test_countdown_number_is_none_outside_countdown():
    g = _game()
    assert g.countdown_number() is None


def test_reset_score():
    g = _game(Move.SCISSORS)
    g.start_round()
    g._phase_started = time.monotonic() - COUNTDOWN_SECONDS
    g.update(Move.ROCK)
    g.reset_score()
    assert g.score.as_tuple() == (0, 0, 0)


@pytest.mark.parametrize("move", list(Move))
def test_ai_provider_pick_returns_valid_move(move):
    assert FixedAI(move).pick() is move
