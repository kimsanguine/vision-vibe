"""HandLite 엔진 테스트 — 카메라 없이 검증 가능한 것만 다룬다.

실제 손 검출 '정확도'는 웹캠이나 실제 손 사진이 있어야 확인할 수 있으므로
여기서는 검증하지 않는다. 대신 아래를 검증한다:
  - 모델 로드/실패 경로
  - 손이 없는 합성 프레임에서 crash 없이 빈 리스트
  - 수명주기(context manager / close / close 후 재사용)
  - VIDEO 모드의 단조 증가 타임스탬프 제약
  - MediaPipe 원시 결과 → 계약 타입(HandResult) 변환의 정확성

마지막 항목은 실제 손 없이도 검증하기 위해 MediaPipe 결과와 같은 모양의
가짜 객체를 변환기에 직접 넣는 방식을 쓴다. '손이 검출됐을 때만 검사'하는
테스트는 합성 프레임에서는 한 번도 실행되지 않아 사실상 아무것도 검증하지
못하기 때문이다.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hand_lite.landmarker import DEFAULT_MODEL_PATH, HandLite
from hand_lite.types import HandResult, Landmark

BLACK_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _noise_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture
def engine():
    """테스트마다 새 엔진 — VIDEO 모드 타임스탬프 상태가 섞이지 않게."""
    with HandLite(num_hands=2) as eng:
        yield eng


# --- 모델 로드 -------------------------------------------------------------

def test_기본_모델_경로가_존재한다():
    assert DEFAULT_MODEL_PATH.is_file(), f"모델이 없습니다: {DEFAULT_MODEL_PATH}"
    assert DEFAULT_MODEL_PATH.is_absolute(), "모델 경로는 절대경로로 해석돼야 한다"


def test_모델_로드_성공(engine):
    assert engine.num_hands == 2
    assert engine.model_path == DEFAULT_MODEL_PATH


def test_잘못된_model_path는_즉시_실패한다(tmp_path):
    missing = tmp_path / "없는모델.task"
    with pytest.raises(FileNotFoundError) as exc:
        HandLite(model_path=str(missing))
    # 조용한 폴백이 아니라 '무엇이 없고 어떻게 고치는지'가 메시지에 있어야 한다.
    message = str(exc.value)
    assert "없는모델.task" in message
    assert "curl" in message, "복구 방법(다운로드 명령)이 메시지에 없다"


def test_디렉토리를_model_path로_주면_실패한다(tmp_path):
    with pytest.raises(FileNotFoundError):
        HandLite(model_path=str(tmp_path))


# --- 손 없는 합성 프레임 ----------------------------------------------------

def test_검은_화면은_빈_리스트를_반환한다(engine):
    assert engine.detect(BLACK_FRAME, timestamp_ms=0) == []


def test_랜덤_노이즈에서도_crash하지_않는다(engine):
    for i in range(3):
        result = engine.detect(_noise_frame(seed=i), timestamp_ms=i * 33)
        assert isinstance(result, list)
        # 노이즈에서 우연히 손이 잡혀도 계약은 지켜져야 한다.
        assert all(isinstance(hand, HandResult) for hand in result)


def test_연속_프레임_처리가_누적_오류없이_동작한다(engine):
    for i in range(10):
        assert engine.detect(BLACK_FRAME, timestamp_ms=i * 33) == []


# --- 입력 검증 -------------------------------------------------------------

def test_None_프레임은_명확한_예외(engine):
    with pytest.raises(ValueError, match="None"):
        engine.detect(None, timestamp_ms=0)


def test_흑백_프레임은_명확한_예외(engine):
    gray = np.zeros((480, 640), dtype=np.uint8)
    with pytest.raises(ValueError, match="3채널"):
        engine.detect(gray, timestamp_ms=0)


# --- VIDEO 모드 타임스탬프 제약 ---------------------------------------------

def test_역행_타임스탬프는_ValueError(engine):
    engine.detect(BLACK_FRAME, timestamp_ms=100)
    with pytest.raises(ValueError, match="단조 증가"):
        engine.detect(BLACK_FRAME, timestamp_ms=50)


def test_동일_타임스탬프도_ValueError(engine):
    """MediaPipe VIDEO 모드는 '같은 값'도 거부한다 — 엄격히 증가해야 한다."""
    engine.detect(BLACK_FRAME, timestamp_ms=100)
    with pytest.raises(ValueError, match="단조 증가"):
        engine.detect(BLACK_FRAME, timestamp_ms=100)


def test_예외_메시지에_실제_타임스탬프값이_들어간다(engine):
    engine.detect(BLACK_FRAME, timestamp_ms=100)
    with pytest.raises(ValueError) as exc:
        engine.detect(BLACK_FRAME, timestamp_ms=42)
    message = str(exc.value)
    assert "100" in message and "42" in message


def test_타임스탬프_위반_후에도_엔진은_계속_쓸_수_있다(engine):
    """위반은 프레임 1장을 거부할 뿐, 엔진을 망가뜨리지 않아야 한다."""
    engine.detect(BLACK_FRAME, timestamp_ms=100)
    with pytest.raises(ValueError):
        engine.detect(BLACK_FRAME, timestamp_ms=50)
    assert engine.detect(BLACK_FRAME, timestamp_ms=200) == []


# --- 수명주기 --------------------------------------------------------------

def test_context_manager가_close를_호출한다():
    with HandLite() as eng:
        assert eng.detect(BLACK_FRAME, timestamp_ms=0) == []
    assert eng._landmarker is None, "__exit__ 이후 리소스가 해제돼야 한다"


def test_예외가_나도_context_manager가_close한다():
    eng_ref = None
    with pytest.raises(RuntimeError):
        with HandLite() as eng:
            eng_ref = eng
            raise RuntimeError("의도적 예외")
    assert eng_ref._landmarker is None


def test_close_후_detect는_RuntimeError():
    """MediaPipe 자체는 close 후 호출을 막지 않는다(해제된 네이티브 자원 접근).
    래퍼가 반드시 막아야 한다.
    """
    eng = HandLite()
    eng.detect(BLACK_FRAME, timestamp_ms=0)
    eng.close()
    with pytest.raises(RuntimeError, match="close"):
        eng.detect(BLACK_FRAME, timestamp_ms=100)


def test_close는_여러번_호출해도_안전하다():
    eng = HandLite()
    eng.close()
    eng.close()
    assert eng._landmarker is None


def test_새_인스턴스는_타임스탬프가_초기화된다():
    """인스턴스마다 시간축이 독립이어야 앱 재시작이 안전하다."""
    with HandLite() as first:
        first.detect(BLACK_FRAME, timestamp_ms=9999)
    with HandLite() as second:
        assert second.detect(BLACK_FRAME, timestamp_ms=0) == []


# --- 계약 타입 변환 ---------------------------------------------------------

def _fake_mp_result(handedness_name: str = "Right", score: float = 0.98):
    """MediaPipe HandLandmarkerResult 와 같은 모양의 가짜 결과.

    실제 손 사진 없이 변환 로직을 검증하기 위한 것. 필드 이름/중첩 구조는
    mediapipe 1.0.0 의 HandLandmarkerResult(handedness: List[List[Category]],
    hand_landmarks: List[List[NormalizedLandmark]]) 를 그대로 따른다.
    """
    landmarks = [
        SimpleNamespace(x=i / 21, y=1 - i / 21, z=-i / 100) for i in range(21)
    ]
    category = SimpleNamespace(category_name=handedness_name, score=score, index=0)
    return SimpleNamespace(hand_landmarks=[landmarks], handedness=[[category]])


def test_변환_결과가_계약_타입을_지킨다():
    hands = HandLite._to_hand_results(_fake_mp_result())
    assert len(hands) == 1
    hand = hands[0]
    assert isinstance(hand, HandResult)
    assert len(hand.landmarks) == 21, "계약: 랜드마크는 항상 21개"
    assert all(isinstance(lm, Landmark) for lm in hand.landmarks)
    assert hand.handedness == "Right"
    assert isinstance(hand.score, float)
    assert 0.0 <= hand.score <= 1.0


def test_변환은_MediaPipe_원시객체를_노출하지_않는다():
    """계약 타입만 밖으로 나가야 한다 — 원시 SimpleNamespace 가 새면 안 된다."""
    hand = HandLite._to_hand_results(_fake_mp_result())[0]
    assert type(hand.landmarks[0]) is Landmark
    assert not isinstance(hand.landmarks[0], SimpleNamespace)


def test_좌표값이_그대로_보존된다():
    hand = HandLite._to_hand_results(_fake_mp_result())[0]
    assert hand.landmarks[0] == Landmark(x=0.0, y=1.0, z=0.0)
    assert hand.landmarks[20].x == pytest.approx(20 / 21)
    assert hand.landmarks[20].z == pytest.approx(-0.20)


@pytest.mark.parametrize("name", ["Left", "Right"])
def test_handedness는_Left_Right_문자열(name):
    hand = HandLite._to_hand_results(_fake_mp_result(handedness_name=name))[0]
    assert hand.handedness == name
    assert isinstance(hand.handedness, str)


def test_손이_없으면_빈_리스트로_변환된다():
    empty = SimpleNamespace(hand_landmarks=[], handedness=[])
    assert HandLite._to_hand_results(empty) == []


def test_양손_결과가_모두_변환된다():
    two = _fake_mp_result()
    two.hand_landmarks = two.hand_landmarks * 2
    two.handedness = [
        [SimpleNamespace(category_name="Left", score=0.9, index=1)],
        [SimpleNamespace(category_name="Right", score=0.95, index=0)],
    ]
    hands = HandLite._to_hand_results(two)
    assert [h.handedness for h in hands] == ["Left", "Right"]
    assert all(len(h.landmarks) == 21 for h in hands)


# --- 실제 검출 정확도(웹캠 필요) --------------------------------------------

def test_실제_손_검출_정확도는_이_환경에서_검증_불가():
    """합성 프레임으로는 검출 '정확도'를 확인할 수 없다는 사실을 명시적으로 남긴다.

    검은 화면·노이즈에서 손이 안 잡히는 건 정상 동작이지 정확도 검증이 아니다.
    실제 손 랜드마크 품질/handedness 정확도는 웹캠으로 app.py 를 돌려야 한다.
    """
    with HandLite() as eng:
        detected = eng.detect(_noise_frame(seed=42), timestamp_ms=0)
    assert detected == [], "노이즈에서 손이 잡히면 오탐 — 별도 조사 필요"
    pytest.skip("실제 손 검출 정확도는 웹캠 필요 — 이 환경에서 미검증")
