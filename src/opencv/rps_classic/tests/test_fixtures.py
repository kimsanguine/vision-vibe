"""실사진 6장 특성화 테스트(characterization test).

성격이 다른 테스트다. 위의 세 파일은 "이렇게 **동작해야 한다**"는 의도를
고정하지만, 이 파일은 "현재 이 파이프라인이 실제로 **이만큼 한다**"는 관측을
고정한다. 목적은 두 가지다.

  1. README 에 적힌 정확도 숫자가 코드와 어긋나지 않게 묶어둔다. 나중에 누가
     파이프라인을 고쳐 정확도가 바뀌면 이 테스트가 **실패해서** README 를
     같이 고치라고 알려준다. (테스트를 고쳐 통과시키는 게 아니라, 문서를
     같이 갱신하는 것이 올바른 대응이다.)
  2. 실패 케이스를 "실패한 채로" 명시적으로 기록한다. 통과하는 것만 테스트하면
     4/6 이 왜 UNKNOWN 인지가 코드베이스 어디에도 남지 않는다.

주의: 여기서 PAPER/SCISSORS 가 UNKNOWN 인 것을 assert 한다고 해서 그게
'올바른 동작'이라는 뜻은 아니다. 이유는 README 의 실패 원인 분석 참조.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from rps_classic.pipeline import detect
from rps_classic.types import Gesture


def _find_workshop_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "yolo" / "rps").is_dir():
            return candidate
    return start.parent


FIXTURE_DIR = _find_workshop_root(Path(__file__).resolve()) / "yolo" / "rps" / "tests" / "fixtures"

# 정답 라벨
TRUTH: dict[str, Gesture] = {
    "rock_0": Gesture.ROCK,
    "rock_1": Gesture.ROCK,
    "paper_0": Gesture.PAPER,
    "paper_1": Gesture.PAPER,
    "scissors_0": Gesture.SCISSORS,
    "scissors_1": Gesture.SCISSORS,
}

# evaluate.py 가 2026-07-29 에 실측한 결과. 하드코딩이 아니라 관측 기록이다.
MEASURED: dict[str, Gesture] = {
    "rock_0": Gesture.ROCK,
    "rock_1": Gesture.ROCK,
    "paper_0": Gesture.UNKNOWN,
    "paper_1": Gesture.UNKNOWN,
    "scissors_0": Gesture.UNKNOWN,
    "scissors_1": Gesture.UNKNOWN,
}
MEASURED_ACCURACY = (2, 6)   # README 의 "2/6" 과 반드시 일치해야 한다

pytestmark = pytest.mark.skipif(
    not FIXTURE_DIR.is_dir(),
    reason=f"픽스처 디렉토리를 찾지 못했다: {FIXTURE_DIR}",
)


def _load(name: str):
    img = cv2.imread(str(FIXTURE_DIR / f"{name}.png"))
    assert img is not None, f"픽스처를 읽지 못했다: {FIXTURE_DIR / name}.png"
    return img


@pytest.mark.parametrize("name", sorted(TRUTH))
def test_모든_픽스처에서_손_컨투어를_찾는다(name):
    """마스크 단계는 6장 모두에서 성공한다 — 실패는 그 뒤에서 일어난다.

    이 구분이 중요하다. shape 가 None 이면 '피부색 검출 실패'이고, shape 는
    있는데 UNKNOWN 이면 '기하/규칙 실패'다. 원인을 뭉뚱그리지 않기 위해
    별도 테스트로 둔다.
    """
    det = detect(_load(name))
    assert det.shape is not None, "피부색 마스크에서 손 크기 덩어리를 찾지 못했다"
    assert det.shape.contour_area > 15000, "손이라기엔 너무 작은 덩어리가 잡혔다"


@pytest.mark.parametrize("name", sorted(TRUTH))
def test_실측된_예측이_재현된다(name):
    """결정적 파이프라인이므로 몇 번을 돌려도 같은 답이 나와야 한다."""
    det = detect(_load(name))
    assert det.gesture is MEASURED[name], (
        f"{name} 의 예측이 바뀌었다. 파이프라인을 의도적으로 고쳤다면 "
        f"evaluate.py 를 다시 돌리고 README 의 정확도 표와 이 파일의 "
        f"MEASURED 를 함께 갱신하라."
    )


def test_실측_정확도가_README와_일치한다():
    n_correct = sum(1 for name, truth in TRUTH.items() if detect(_load(name)).gesture is truth)
    assert (n_correct, len(TRUTH)) == MEASURED_ACCURACY, (
        f"정확도가 {n_correct}/{len(TRUTH)} 로 바뀌었다. README 의 3자 비교표를 "
        f"함께 갱신해야 한다 — 문서와 코드가 어긋난 채로 두지 않는다."
    )


def test_바위는_두_장_모두_맞춘다():
    """전통 기법이 실제로 잘하는 케이스 — 특징 없는 덩어리 구분.

    이 테스트가 깨지면 파이프라인이 근본적으로 망가진 것이다(2/6 조차 못 낸다).
    """
    for name in ("rock_0", "rock_1"):
        det = detect(_load(name))
        assert det.gesture is Gesture.ROCK
        assert det.shape.finger_gaps == 0
        assert det.shape.solidity > 0.85, "주먹은 볼록 껍질을 거의 꽉 채워야 한다"


def test_가위는_골을_정확히_2개_잡지만_규칙이_틀려_UNKNOWN이_된다():
    """실패의 성격을 코드로 못박는다 — '검출 실패'가 아니라 '규칙 불일치'다.

    이 픽스처의 가위 포즈는 엄지가 펴져 있어 펴진 손가락이 실제로 3개다.
    골 2개(엄지-검지, 검지-중지)는 기하학적으로 **정답**이고, 틀린 것은
    '가위 = 손가락 2개'라는 매핑이다. convexity defect 에는 어느 골이 엄지
    쪽인지 알 정보가 없어 이 파이프라인 안에서는 해결 경로가 없다.
    """
    for name in ("scissors_0", "scissors_1"):
        det = detect(_load(name))
        assert det.shape.finger_gaps == 2, "골 검출 자체는 성공한다"
        assert det.shape.fingers == 3, "엄지까지 세어 3개가 된다"
        assert det.gesture is Gesture.UNKNOWN, "3개는 셋 중 어느 것도 아니므로 기권한다"


def test_틀린_경우에도_억지로_다른_손모양을_내지_않는다():
    """오답이 UNKNOWN 이지 ROCK/PAPER/SCISSORS 오분류가 아니라는 점을 고정한다.

    이 성질은 게임에서 실질적으로 중요하다. 기권은 다시 내면 되지만, 확신에 찬
    오분류는 사용자가 진 이유를 납득할 수 없게 만든다.
    """
    for name, truth in TRUTH.items():
        det = detect(_load(name))
        if det.gesture is not truth:
            assert det.gesture is Gesture.UNKNOWN, (
                f"{name}: {truth.name} 을 {det.gesture.name} 로 **오분류**했다. "
                f"기권(UNKNOWN)이 아닌 오분류는 이 파이프라인의 설계 원칙 위반이다."
            )
