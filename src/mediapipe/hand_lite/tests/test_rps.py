"""hand_lite.rps 검증 — 순수 로직이라 카메라/MediaPipe 없이 100% 테스트 가능하다.

승부 판정 전 조합(3x3 + UNKNOWN), AI 패 결정성, 프레임 흔들림 안정화 로직을
각각 검증한다. "결과가 WIN이다"라는 사실이 아니라 "왜 그래야 하는가"(가위바위보
규칙, 확신 없는 입력은 억지로 분류하지 않는다, 지터를 다수결로 걸러낸다)를
검증하는 것이 목표다.
"""

from __future__ import annotations

import pytest

from hand_lite.rps import AIMoveProvider, Game, MoveStabilizer, Result, Score, judge
from hand_lite.types import Gesture

ROCK, PAPER, SCISSORS, UNKNOWN = Gesture.ROCK, Gesture.PAPER, Gesture.SCISSORS, Gesture.UNKNOWN


class TestJudgeAllPlayableCombinations:
    """3x3 = 9가지 조합 전부 — 표준 가위바위보 규칙(가위<보<바위<가위)을 그대로."""

    @pytest.mark.parametrize(
        "player, ai, expected",
        [
            (ROCK, ROCK, Result.DRAW),
            (ROCK, PAPER, Result.LOSE),       # 보가 바위를 이긴다
            (ROCK, SCISSORS, Result.WIN),     # 바위가 가위를 이긴다
            (PAPER, ROCK, Result.WIN),        # 보가 바위를 이긴다
            (PAPER, PAPER, Result.DRAW),
            (PAPER, SCISSORS, Result.LOSE),   # 가위가 보를 이긴다
            (SCISSORS, ROCK, Result.LOSE),    # 바위가 가위를 이긴다
            (SCISSORS, PAPER, Result.WIN),    # 가위가 보를 이긴다
            (SCISSORS, SCISSORS, Result.DRAW),
        ],
    )
    def test_combination(self, player, ai, expected):
        assert judge(player, ai) is expected


class TestJudgeUnknownIsNeverForced:
    """UNKNOWN이 끼면 승부를 내지 않는다 — 확신 없는 입력을 억지로 분류하지 않는다."""

    @pytest.mark.parametrize("ai", [ROCK, PAPER, SCISSORS, UNKNOWN])
    def test_player_unknown_is_always_undecidable(self, ai):
        assert judge(UNKNOWN, ai) is Result.UNDECIDABLE

    @pytest.mark.parametrize("player", [ROCK, PAPER, SCISSORS])
    def test_ai_unknown_is_undecidable_even_if_player_is_valid(self, player):
        """실제로는 AIMoveProvider가 UNKNOWN을 고르지 않지만, 방어적으로
        플레이어 쪽 규칙과 대칭이어야 한다."""
        assert judge(player, UNKNOWN) is Result.UNDECIDABLE


class TestAIMoveProviderDeterminism:
    def test_same_seed_produces_same_sequence(self):
        a = AIMoveProvider(seed=42)
        b = AIMoveProvider(seed=42)
        sequence_a = [a.pick() for _ in range(20)]
        sequence_b = [b.pick() for _ in range(20)]
        assert sequence_a == sequence_b

    def test_different_seeds_can_diverge(self):
        """같은 시드가 아니면 반드시 다르다고 단정할 수는 없지만(가위바위보는
        선택지가 3개뿐이라 우연히 같을 수 있다), 20판 전부가 우연히 일치할
        확률은 3^-20으로 사실상 0이라 회귀 검증으로 충분히 안전하다."""
        a = AIMoveProvider(seed=1)
        b = AIMoveProvider(seed=2)
        sequence_a = [a.pick() for _ in range(20)]
        sequence_b = [b.pick() for _ in range(20)]
        assert sequence_a != sequence_b

    def test_pick_never_returns_unknown(self):
        provider = AIMoveProvider(seed=7)
        picks = {provider.pick() for _ in range(50)}
        assert UNKNOWN not in picks
        assert picks <= {ROCK, PAPER, SCISSORS}


class TestMoveStabilizer:
    """지터(프레임 간 흔들림) 제거 — 최근 window개 중 min_votes 이상이어야 확정."""

    def test_empty_buffer_is_not_stable(self):
        stabilizer = MoveStabilizer()
        assert stabilizer.stable_gesture() is None

    def test_below_min_votes_is_not_stable(self):
        stabilizer = MoveStabilizer(window=7, min_votes=4)
        for _ in range(3):  # min_votes(4)에 못 미침
            stabilizer.update(ROCK)
        assert stabilizer.stable_gesture() is None

    def test_reaching_min_votes_becomes_stable(self):
        stabilizer = MoveStabilizer(window=7, min_votes=4)
        for _ in range(4):
            stabilizer.update(ROCK)
        assert stabilizer.stable_gesture() is ROCK

    def test_majority_wins_even_when_interleaved_with_jitter(self):
        """손을 바꾸는 순간처럼 프레임 사이사이 다른 값이 섞여도, 과반을 채운
        쪽이 확정된다 — 순서가 아니라 누적 표 개수만 본다."""
        stabilizer = MoveStabilizer(window=7, min_votes=4)
        for gesture in [ROCK, PAPER, ROCK, PAPER, ROCK, PAPER, ROCK]:
            stabilizer.update(gesture)
        # 7표 중 ROCK 4, PAPER 3 — ROCK이 min_votes(4)에 도달해 확정된다.
        assert stabilizer.stable_gesture() is ROCK

    def test_true_jitter_with_no_majority_stays_none(self):
        stabilizer = MoveStabilizer(window=6, min_votes=4)
        for gesture in [ROCK, PAPER, ROCK, PAPER, ROCK, PAPER]:
            stabilizer.update(gesture)
        # 3 vs 3 — 어느 쪽도 min_votes(4)에 못 미친다.
        assert stabilizer.stable_gesture() is None

    def test_old_frames_fall_out_of_window(self):
        """window를 넘는 과거 프레임은 버려진다 — deque(maxlen)이 자동 처리."""
        stabilizer = MoveStabilizer(window=4, min_votes=3)
        for _ in range(4):
            stabilizer.update(ROCK)
        assert stabilizer.stable_gesture() is ROCK
        for _ in range(4):
            stabilizer.update(PAPER)
        # ROCK 표는 전부 window 밖으로 밀려났다.
        assert stabilizer.stable_gesture() is PAPER

    def test_unknown_can_become_stable(self):
        """손이 계속 안 잡히거나 애매한 상태도 '안정된 상태'로 인정한다 —
        judge()가 이후 UNDECIDABLE로 처리하는 것과 별개로, 안정화 자체는
        UNKNOWN을 차별하지 않는다."""
        stabilizer = MoveStabilizer(window=5, min_votes=3)
        for _ in range(3):
            stabilizer.update(UNKNOWN)
        assert stabilizer.stable_gesture() is UNKNOWN

    def test_clear_resets_buffer(self):
        stabilizer = MoveStabilizer(window=5, min_votes=3)
        for _ in range(3):
            stabilizer.update(ROCK)
        assert stabilizer.stable_gesture() is ROCK
        stabilizer.clear()
        assert stabilizer.stable_gesture() is None

    def test_min_votes_greater_than_window_is_rejected(self):
        """min_votes가 window보다 크면 영원히 확정될 수 없는 죽은 설정이다 —
        조용히 허용하지 않고 생성 시점에 막는다."""
        with pytest.raises(ValueError):
            MoveStabilizer(window=3, min_votes=4)


class TestScore:
    def test_win_lose_draw_increments_correct_counter(self):
        score = Score()
        score.record(Result.WIN)
        score.record(Result.WIN)
        score.record(Result.LOSE)
        score.record(Result.DRAW)
        assert score.as_tuple() == (2, 1, 1)

    def test_undecidable_does_not_change_score(self):
        score = Score()
        score.record(Result.UNDECIDABLE)
        assert score.as_tuple() == (0, 0, 0)


class TestGamePlayRound:
    """Game은 안정화된 손모양 1개를 받아 라운드 1회를 즉시 판정하는 최소 상태 객체다."""

    def test_valid_gesture_produces_scored_round(self):
        # seed=0의 첫 pick이 무엇인지는 AIMoveProvider 자체 테스트에서 이미
        # 결정적임을 증명했으니, 여기서는 "AI가 무엇을 냈든 judge() 규칙과
        # 일치하는 결과가 나오고 점수에 반영된다"만 검증한다.
        game = Game(ai_provider=AIMoveProvider(seed=0))
        round_state = game.play_round(ROCK)

        assert round_state.player is ROCK
        assert round_state.ai in (ROCK, PAPER, SCISSORS)
        assert round_state.result is judge(ROCK, round_state.ai)
        assert game.last_round is round_state
        assert sum(game.score.as_tuple()) == 1  # 승/패/무 중 정확히 하나만 기록됨

    def test_unknown_gesture_produces_undecidable_round_without_scoring(self):
        game = Game(ai_provider=AIMoveProvider(seed=0))
        round_state = game.play_round(UNKNOWN)

        assert round_state.player is UNKNOWN
        assert round_state.ai is UNKNOWN
        assert round_state.result is Result.UNDECIDABLE
        assert game.score.as_tuple() == (0, 0, 0)

    def test_reset_score_zeroes_out_but_keeps_last_round(self):
        game = Game(ai_provider=AIMoveProvider(seed=0))
        game.play_round(ROCK)
        assert sum(game.score.as_tuple()) == 1

        game.reset_score()
        assert game.score.as_tuple() == (0, 0, 0)
        assert game.last_round is not None  # 점수만 초기화, 직전 라운드 기록은 유지
