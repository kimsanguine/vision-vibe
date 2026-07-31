"""모델 로드·추론 검증 (웹캠 없이 정지 이미지·영상 파일로).

웹캠 실캡처는 이 환경에서 검증 불가능하므로, 동일 코드 경로를 타는
영상 파일 프레임으로 대체 검증한다.
"""

from pathlib import Path

import cv2
import pytest

from rps.detector import MoveStabilizer, RPSDetector
from rps.judge import Move

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = PROJECT_ROOT / "models" / "rps_yolo11x_leeyunjai.pt"
VIDEO = PROJECT_ROOT / "tests" / "fixtures" / "rps_video.avi"

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="모델 파일 없음")


@pytest.fixture(scope="module")
def detector() -> RPSDetector:
    return RPSDetector(MODEL, conf=0.25)


def test_model_loads_with_three_rps_classes(detector):
    names = {n.lower() for n in detector.class_names.values()}
    assert names == {"rock", "paper", "scissors"}


def test_predict_runs_without_error_on_blank_frame(detector):
    import numpy as np

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect(blank) is None  # 손이 없으면 검출 없음


@pytest.mark.skipif(not VIDEO.exists(), reason="테스트 영상 없음")
def test_detects_moves_across_video_frames(detector):
    """실제 손 영상에서 세 손모양이 모두 한 번 이상 검출되어야 한다."""
    cap = cv2.VideoCapture(str(VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    found: set[Move] = set()
    detected_frames = 0
    sampled = 0
    for idx in range(0, total, max(1, total // 20)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        sampled += 1
        det = detector.detect(frame)
        if det:
            detected_frames += 1
            found.add(det.move)
            assert 0.0 < det.confidence <= 1.0
            x1, y1, x2, y2 = det.box
            assert x2 > x1 and y2 > y1
    cap.release()

    assert sampled > 0
    assert detected_frames / sampled >= 0.7, f"검출률 미달: {detected_frames}/{sampled}"
    assert found == set(Move), f"검출되지 않은 손모양: {set(Move) - found}"


# --- MoveStabilizer ---

def test_stabilizer_needs_enough_votes():
    s = MoveStabilizer(window=7, min_votes=4)
    for _ in range(3):
        s.update(Move.ROCK)
    assert s.stable_move() is None
    s.update(Move.ROCK)
    assert s.stable_move() is Move.ROCK


def test_stabilizer_ignores_single_frame_noise():
    s = MoveStabilizer(window=7, min_votes=4)
    for move in [Move.ROCK, Move.ROCK, Move.PAPER, Move.ROCK, Move.ROCK, Move.SCISSORS]:
        s.update(move)
    assert s.stable_move() is Move.ROCK


def test_stabilizer_returns_none_when_no_detection():
    s = MoveStabilizer()
    for _ in range(7):
        s.update(None)
    assert s.stable_move() is None


def test_stabilizer_clear():
    s = MoveStabilizer(window=7, min_votes=4)
    for _ in range(5):
        s.update(Move.PAPER)
    s.clear()
    assert s.stable_move() is None
