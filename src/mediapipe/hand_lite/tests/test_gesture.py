"""gesture.py 판정 로직 검증.

이 모듈은 순수 함수 판정 레이어이므로 실제 카메라/MediaPipe 없이 합성
랜드마크(21점)만으로 대부분 테스트 가능하다. 각 테스트는 "좌표가 이렇게
나오면 왜 그렇게 판정되어야 하는가"를 검증한다 — classify가 ROCK을
반환한다는 사실이 아니라, "다섯 손가락이 모두 굽었을 때"라는 조건이 핵심이다.

--- 실사진 검증으로 드러난 교훈 ---
최초 버전은 엄지를 "x축 tip vs IP + handedness 반전"으로 판정했고, 합성
fixture는 전부 통과했다. 하지만 `../../yolo/rps/tests/fixtures/`의 실제 손
사진 6장으로 돌려보니 엄지 정확도 0/6 — 검지~소지(y축 규칙)는 6/6 정확했다.
원인은 ① 사진이 손등이 보이는 각도였고 ② 주먹에서 엄지가 접힌 손가락 위에
얹혀 x좌표만으로는 "펴짐"으로 잘못 읽혔기 때문이다. 합성 fixture가 전부
통과한다고 실제 손에서 작동한다는 뜻이 아니라는 걸 이 프로젝트 스스로
증명한 셈이다. 그래서 TestRealPhotoRegression으로 실사진 end-to-end
회귀를 별도로 고정한다 — 합성 fixture만으로는 이런 결함이 다시 통과해도
잡을 수 없다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from hand_lite.gesture import classify, fingers_up
from hand_lite.landmarker import HandLite
from hand_lite.types import Gesture, HandResult, Landmark

# 손가락별 (MCP, PIP, DIP, TIP) 인덱스 — types.py 의 MediaPipe 21점 규약과 동일.
_INDEX = (5, 6, 7, 8)
_MIDDLE = (9, 10, 11, 12)
_RING = (13, 14, 15, 16)
_PINKY = (17, 18, 19, 20)


def _finger_points(base_x: float, mcp_y: float, extended: bool):
    """검지~소지 한 손가락의 4관절 좌표.

    펴짐: tip.y < pip.y (관절을 따라 y가 계속 작아짐 = 화면 위쪽으로 쭉 뻗음).
    굽음: tip.y >= pip.y (손끝이 손바닥 쪽으로 말려 pip보다 아래/같은 높이).
    """
    mcp = Landmark(x=base_x, y=mcp_y)
    if extended:
        pip = Landmark(x=base_x, y=mcp_y - 0.05)
        dip = Landmark(x=base_x, y=mcp_y - 0.10)
        tip = Landmark(x=base_x, y=mcp_y - 0.15)
    else:
        pip = Landmark(x=base_x, y=mcp_y - 0.02)
        dip = Landmark(x=base_x, y=mcp_y - 0.01)
        tip = Landmark(x=base_x, y=mcp_y)
    return mcp, pip, dip, tip


def _thumb_points(extended: bool):
    """엄지는 "검지 밑동(index_mcp, x=0.4, y=0.6)에서 얼마나 벌어졌는가"로
    판정한다(거리비 방식) — x/y 방향이나 handedness와 무관하다.

    펴짐: tip을 검지 밑동에서 멀리 떨어뜨린다(ratio ≈ 1.0 > 임계 0.75).
    굽음: tip을 검지 밑동 바로 옆에 둔다 — 실제 주먹에서 엄지가 접힌
    손가락 위에 얹히는 모양을 흉내(ratio ≈ 0.08 < 0.75).
    """
    cmc = Landmark(x=0.5, y=0.85)
    mcp = Landmark(x=0.5, y=0.78)
    ip = Landmark(x=0.5, y=0.70)
    if extended:
        tip = Landmark(x=0.75, y=0.60)
    else:
        tip = Landmark(x=0.42, y=0.62)
    return cmc, mcp, ip, tip


def build_hand(
    *,
    thumb: bool,
    index: bool,
    middle: bool,
    ring: bool,
    pinky: bool,
    handedness: str = "Right",
) -> HandResult:
    """21개 랜드마크를 손으로 조립해 지정한 손가락 펴짐 조합을 표현한다.

    handedness는 HandResult 계약상 필수 필드라 받지만, 현재 gesture.py의
    판정 로직은 이 값을 전혀 참조하지 않는다(거리비 방식으로 바뀌면서
    방향 의존이 사라짐) — 이 사실은 TestFingersUpIgnoresHandedness가 고정한다.
    """
    landmarks: list[Landmark] = [Landmark(x=0.5, y=0.95)] * 21  # 자리 확보용 기본값

    wrist = Landmark(x=0.5, y=0.95)
    thumb_pts = _thumb_points(thumb)
    index_pts = _finger_points(0.4, 0.6, index)
    middle_pts = _finger_points(0.5, 0.6, middle)
    ring_pts = _finger_points(0.6, 0.6, ring)
    pinky_pts = _finger_points(0.7, 0.65, pinky)

    landmarks[0] = wrist
    landmarks[1:5] = thumb_pts
    landmarks[5:9] = index_pts
    landmarks[9:13] = middle_pts
    landmarks[13:17] = ring_pts
    landmarks[17:21] = pinky_pts

    return HandResult(landmarks=landmarks, handedness=handedness, score=0.95)


class TestFingersUpIndexToPinky:
    def test_extended_finger_has_tip_above_pip(self):
        """검지만 편 손 → 검지 슬롯만 True, 나머지는 False."""
        hand = build_hand(thumb=False, index=True, middle=False, ring=False, pinky=False)
        _, index, middle, ring, pinky = fingers_up(hand)
        assert index is True
        assert middle is False
        assert ring is False
        assert pinky is False

    def test_all_four_fingers_extended(self):
        hand = build_hand(thumb=False, index=True, middle=True, ring=True, pinky=True)
        _, index, middle, ring, pinky = fingers_up(hand)
        assert (index, middle, ring, pinky) == (True, True, True, True)

    def test_all_four_fingers_folded(self):
        hand = build_hand(thumb=False, index=False, middle=False, ring=False, pinky=False)
        _, index, middle, ring, pinky = fingers_up(hand)
        assert (index, middle, ring, pinky) == (False, False, False, False)


class TestThumbSpreadRatio:
    """엄지는 '검지 밑동에서 얼마나 벌어졌는가'의 거리비로 판정한다
    (실사진 검증 후 x축+handedness 방식에서 교체됨)."""

    def test_thumb_far_from_index_mcp_is_extended(self):
        hand = build_hand(thumb=True, index=False, middle=False, ring=False, pinky=False)
        thumb, *_ = fingers_up(hand)
        assert thumb is True

    def test_thumb_close_to_index_mcp_is_folded(self):
        """실제 주먹에서 엄지가 접힌 손가락 위에 얹히는 모양 — 검지 밑동과
        가까운 위치. x좌표만 보던 예전 규칙은 이 케이스를 '펴짐'으로
        오판했었다."""
        hand = build_hand(thumb=False, index=False, middle=False, ring=False, pinky=False)
        thumb, *_ = fingers_up(hand)
        assert thumb is False


class TestFingersUpIgnoresHandedness:
    """거리비 방식은 방향(x/y 부호)이 아니라 거리만 보므로 handedness가
    필요 없다. fingers_up이 hand.handedness를 실제로 안 쓰는지, 즉 같은
    좌표에 Left/Right를 넣어도 결과가 똑같은지를 직접 고정한다."""

    def test_same_landmarks_give_same_result_regardless_of_handedness(self):
        landmarks = [Landmark(x=0.5, y=0.95)] * 21
        landmarks[0] = Landmark(x=0.5, y=0.95)
        landmarks[1:5] = _thumb_points(extended=True)
        landmarks[5:9] = _finger_points(0.4, 0.6, True)
        landmarks[9:13] = _finger_points(0.5, 0.6, False)
        landmarks[13:17] = _finger_points(0.6, 0.6, False)
        landmarks[17:21] = _finger_points(0.7, 0.65, False)

        right_hand = HandResult(landmarks=landmarks, handedness="Right", score=0.9)
        left_hand = HandResult(landmarks=landmarks, handedness="Left", score=0.9)

        assert fingers_up(right_hand) == fingers_up(left_hand)


class TestFingersUpBoundary:
    def test_raises_when_landmark_count_is_not_21(self):
        hand = HandResult(landmarks=[Landmark(x=0.5, y=0.5)] * 20, handedness="Right", score=0.9)
        with pytest.raises(ValueError):
            fingers_up(hand)


class TestClassifyRockPaperScissors:
    def test_fist_with_all_fingers_folded_is_rock(self):
        hand = build_hand(thumb=False, index=False, middle=False, ring=False, pinky=False)
        assert classify(hand) is Gesture.ROCK

    def test_open_hand_with_all_five_extended_is_paper(self):
        hand = build_hand(thumb=True, index=True, middle=True, ring=True, pinky=True)
        assert classify(hand) is Gesture.PAPER

    def test_index_and_middle_with_thumb_extended_is_scissors(self):
        """실제 사람은 가위를 낼 때 엄지를 벌린다(scissors_0/1.png 실측) —
        엄지가 펴져 있어도 가위로 판정되어야 한다."""
        hand = build_hand(thumb=True, index=True, middle=True, ring=False, pinky=False)
        assert classify(hand) is Gesture.SCISSORS

    def test_index_and_middle_with_thumb_folded_is_also_scissors(self):
        """엄지가 접혀 있어도(사람마다 습관 차이) 검지+중지 조건만 맞으면
        가위 — 가위 판정에 엄지 상태는 관여하지 않는다."""
        hand = build_hand(thumb=False, index=True, middle=True, ring=False, pinky=False)
        assert classify(hand) is Gesture.SCISSORS

    def test_rock_paper_scissors_hold_regardless_of_handedness_label(self):
        """엄지 판정이 거리비 방식으로 바뀌어 handedness 의존이 없어졌으니
        Left로 라벨링해도 동일하게 판정되어야 한다."""
        rock = build_hand(thumb=False, index=False, middle=False, ring=False, pinky=False, handedness="Left")
        paper = build_hand(thumb=True, index=True, middle=True, ring=True, pinky=True, handedness="Left")
        scissors = build_hand(thumb=True, index=True, middle=True, ring=False, pinky=False, handedness="Left")

        assert classify(rock) is Gesture.ROCK
        assert classify(paper) is Gesture.PAPER
        assert classify(scissors) is Gesture.SCISSORS


class TestClassifyUnknownForAmbiguousShapes:
    def test_thumb_and_index_only_is_unknown(self):
        """가위바위보 세 형태 어디에도 안 맞는 조합 — 억지로 분류하지 않는다."""
        hand = build_hand(thumb=True, index=True, middle=False, ring=False, pinky=False)
        assert classify(hand) is Gesture.UNKNOWN

    def test_index_middle_ring_is_unknown_not_scissors(self):
        """가위는 약지+소지가 굽어 있어야 한다 — 약지까지 펴지면 가위가 아니다."""
        hand = build_hand(thumb=False, index=True, middle=True, ring=True, pinky=False)
        assert classify(hand) is Gesture.UNKNOWN

    def test_only_pinky_is_unknown(self):
        hand = build_hand(thumb=False, index=False, middle=False, ring=False, pinky=True)
        assert classify(hand) is Gesture.UNKNOWN

    def test_wrong_landmark_count_is_unknown_not_a_crash(self):
        """랜드마크가 21개가 아닌 손상된 검출 결과가 들어와도 앱을 죽이지
        않고 UNKNOWN으로 처리한다 (fingers_up의 ValueError를 classify가 흡수)."""
        hand = HandResult(landmarks=[Landmark(x=0.5, y=0.5)] * 5, handedness="Right", score=0.9)
        assert classify(hand) is Gesture.UNKNOWN


# --- 실사진 end-to-end 회귀 --------------------------------------------------
# 옆 프로젝트(yolo/rps)의 기존 테스트 픽스처를 그대로 재사용한다. 이 파일이
# 없는 환경(다른 머신, CI)에서는 skip — 카메라를 열지 않고, 픽스처도 만들지
# 않는다.
def _find_workshop_root(start: Path) -> Path:
    """yolo/rps 가 있는 조상 디렉토리를 찾는다.

    `.parent.parent.parent` 로 고정하면 이 테스트를 worktree
    (.worktrees/rps-game/tests/) 처럼 한 단계 깊은 곳에서 돌릴 때
    `.worktrees/yolo/rps` 를 가리켜 픽스처를 못 찾는다. 그러면 실사진
    회귀 테스트 6개가 **조용히 skip** 되어, 엄지 결함이 재발해도 초록불이
    뜬다. 실행 깊이와 무관하게 위로 탐색한다(bench.py 와 같은 방식).
    """
    for candidate in (start, *start.parents):
        if (candidate / "yolo" / "rps").is_dir():
            return candidate
    return start.parent.parent


FIXTURES_DIR = _find_workshop_root(Path(__file__).resolve().parent) / "yolo" / "rps" / "tests" / "fixtures"

_REAL_PHOTO_CASES = [
    ("rock_0.png", Gesture.ROCK),
    ("rock_1.png", Gesture.ROCK),
    ("paper_0.png", Gesture.PAPER),
    ("paper_1.png", Gesture.PAPER),
    ("scissors_0.png", Gesture.SCISSORS),
    ("scissors_1.png", Gesture.SCISSORS),
]


@pytest.mark.skipif(
    not FIXTURES_DIR.is_dir(),
    reason=f"실사진 픽스처 없음: {FIXTURES_DIR} (yolo/rps 프로젝트가 없는 환경)",
)
class TestRealPhotoRegression:
    """합성 fixture 위 테스트들은 전부 통과했지만 엄지 판정은 실제 사진
    6장 중 0장이 맞았다 — 이 결함이 다시 조용히 통과하지 않도록 실사진으로
    end-to-end 회귀를 고정한다. 카메라는 열지 않는다(정적 이미지 파일만 사용).
    """

    @pytest.mark.parametrize("filename, expected", _REAL_PHOTO_CASES)
    def test_real_photo_classifies_correctly(self, filename, expected):
        image_path = FIXTURES_DIR / filename
        frame = cv2.imread(str(image_path))
        assert frame is not None, f"이미지를 못 읽었습니다: {image_path}"

        # 테스트마다 새 엔진 — VIDEO 모드 타임스탬프 상태를 공유하지 않기 위함.
        with HandLite(num_hands=1) as engine:
            hands = engine.detect(frame, timestamp_ms=0)

        assert len(hands) == 1, f"{filename}에서 손이 정확히 1개 검출되어야 함(검출={len(hands)}개)"
        assert classify(hands[0]) is expected, f"{filename}: fingers_up={fingers_up(hands[0])}"
