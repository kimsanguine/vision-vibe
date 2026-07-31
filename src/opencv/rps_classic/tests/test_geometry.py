"""기하 계산 검증 — 정답이 수학적으로 확정되는 부분만 합성 데이터로 고정한다.

이 파일이 테스트하는 것은 "코드가 도는가"가 아니라 "각도 계산이 **맞는가**"다.
그래서 실제 사진을 쓰지 않는다 — 사진에는 정답 각도가 없다. 직각삼각형·정삼각형
처럼 손으로 계산 가능한 좌표를 넣어 기대값과 대조한다.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from rps_classic.fingers import (
    convexity_defects,
    count_finger_gaps,
    defect_angle_deg,
    largest_contour,
)


# ── 코사인 법칙 각도 계산 ────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("start", "end", "far", "expected"),
    [
        # far 를 원점에 두고 두 변을 x축·y축에 올리면 정확히 90도.
        ((10, 0), (0, 10), (0, 0), 90.0),
        # 한 변 길이를 바꿔도 두 변이 직교하면 여전히 90도(길이 무관 검증).
        ((100, 0), (0, 3), (0, 0), 90.0),
        # 정삼각형: 모든 내각 60도.
        ((0, 0), (2, 0), (1, math.sqrt(3)), 60.0),
        # 두 변이 겹치면(같은 방향) 0도.
        ((10, 0), (5, 0), (0, 0), 0.0),
        # far 가 start-end 선분 위 중간에 있으면 완전히 펴진 180도.
        ((-5, 0), (5, 0), (0, 0), 180.0),
        # 30-60-90 직각삼각형 A(0,0) B(3,0) C(0,√3) 의 B 꼭짓점 = 30도.
        ((0, 0), (0, math.sqrt(3)), (3, 0), 30.0),
        # 같은 삼각형의 C 꼭짓점 = 60도.
        ((0, 0), (3, 0), (0, math.sqrt(3)), 60.0),
    ],
)
def test_각도가_알려진_삼각형에서_정확히_계산된다(start, end, far, expected):
    assert defect_angle_deg(start, end, far) == pytest.approx(expected, abs=1e-6)


def test_start와_end를_바꿔도_각도가_같다():
    """far 에서 본 사잇각은 두 변의 순서와 무관하다 — 대칭성."""
    a = defect_angle_deg((3, 7), (11, 2), (5, 5))
    b = defect_angle_deg((11, 2), (3, 7), (5, 5))
    assert a == pytest.approx(b, abs=1e-9)


def test_평행이동해도_각도가_같다():
    """각도는 절대 좌표가 아니라 상대 배치의 함수다."""
    base = defect_angle_deg((10, 0), (0, 10), (0, 0))
    moved = defect_angle_deg((1010, 500), (1000, 510), (1000, 500))
    assert base == pytest.approx(moved, abs=1e-9)


@pytest.mark.parametrize(
    ("start", "end", "far"),
    [
        ((0, 0), (5, 5), (0, 0)),   # far == start
        ((0, 0), (5, 5), (5, 5)),   # far == end
        ((3, 3), (3, 3), (3, 3)),   # 세 점이 모두 같음
    ],
)
def test_각이_정의되지_않으면_180도를_반환한다(start, end, far):
    """0도(가장 뾰족함)가 아니라 180도(가장 둔함)로 처리해야 한다.

    이 값은 "90도 이하면 손가락 사이로 인정"에 그대로 쓰인다. 정의되지 않는
    입력을 0도로 돌려주면 존재하지 않는 손가락 골이 하나 생겨 손가락 수가
    부풀려진다. 판정 불가는 반드시 '거절' 쪽으로 보내야 한다.
    """
    assert defect_angle_deg(start, end, far) == 180.0


def test_각도는_항상_0에서_180_사이다():
    """부동소수점 오차로 acos 입력이 [-1,1] 를 벗어나도 NaN 이 나오면 안 된다."""
    rng = np.random.default_rng(20260729)
    for _ in range(500):
        pts = rng.uniform(-1000, 1000, size=(3, 2))
        angle = defect_angle_deg(tuple(pts[0]), tuple(pts[1]), tuple(pts[2]))
        assert not math.isnan(angle)
        assert 0.0 <= angle <= 180.0


def test_아주_가까운_두_점도_NaN을_내지_않는다():
    """b*c 가 0은 아니지만 극도로 작을 때 cos 이 1을 살짝 넘는 경우."""
    angle = defect_angle_deg((1e-9, 0.0), (2e-9, 0.0), (0.0, 0.0))
    assert not math.isnan(angle)
    assert 0.0 <= angle <= 180.0


# ── 합성 손 실루엣으로 골 개수 검증 ──────────────────────────────────────

def make_synthetic_hand(n_fingers: int, *, size: int = 400, arc: int = 8) -> np.ndarray:
    """손바닥 사각형 + 손가락 n개로 만든 이진 실루엣.

    실제 사진 대신 합성 도형을 쓰는 이유: "손가락 5개를 폈다"는 정답을 코드가
    **정의**할 수 있어야 골 개수 계산의 정오를 판정할 수 있다. 사진에는 그
    정답이 없다(사람이 라벨을 붙여도 골의 기하학적 개수와 일치한다는 보장이 없다).

    arc 는 손끝 높이의 곡률이다. 손끝 y = 기준 + arc*(i - 가운데)^2 이므로
    **가운데 손가락이 가장 길고 바깥으로 갈수록 짧아지는** 실제 손 모양이 된다.
    arc=0 이면 손끝이 전부 같은 높이의 일직선이 되는데, 그 경우 골 검출이
    무너진다(아래 test_손끝이_일직선이면_골_검출이_무너진다 참조). 기본값을
    0이 아니라 8로 둔 것은 그래서다 — 평평한 손은 이 기법의 퇴화 케이스다.

    손가락 N개 → 손가락 사이 골 N-1개 가 기대값이다.
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    palm_top, palm_bottom = 250, 390
    finger_w, gap_w, base_top = 24, 26, 120

    span = n_fingers * finger_w + max(0, n_fingers - 1) * gap_w
    x0 = (size - span) // 2
    # 손바닥은 손가락 전체 폭보다 넉넉히 크게 — 손가락이 손바닥 밖으로 나가면
    # 실루엣이 분리돼 컨투어가 여러 개가 된다.
    cv2.rectangle(mask, (x0 - 20, palm_top), (x0 + span + 20, palm_bottom), 255, -1)

    center = (n_fingers - 1) / 2
    for i in range(n_fingers):
        fx = x0 + i * (finger_w + gap_w)
        top = base_top + int(arc * (i - center) ** 2)
        cv2.rectangle(mask, (fx, top), (fx + finger_w, palm_top), 255, -1)
        # 둥근 손끝. 실제 손가락 끝은 사각이 아니라 반원에 가깝고, 이 곡률이
        # 있어야 손끝이 볼록 껍질의 '정점'으로 확실히 잡힌다.
        cv2.circle(mask, (fx + finger_w // 2, top), finger_w // 2, 255, -1)
    return mask


@pytest.mark.parametrize("n_fingers", [2, 3, 4, 5])
def test_합성_손에서_골_개수는_손가락수_빼기_1이다(n_fingers):
    mask = make_synthetic_hand(n_fingers)
    contour = largest_contour(mask)
    assert contour is not None, "합성 실루엣에서 컨투어를 찾지 못했다"
    gaps = count_finger_gaps(contour, convexity_defects(contour))
    assert gaps == n_fingers - 1


@pytest.mark.parametrize("n_fingers", [3, 4, 5])
def test_손끝이_일직선이면_골_검출이_무너진다(n_fingers):
    """이 기법의 퇴화 케이스를 명시적으로 기록한다 — 버그가 아니라 원리적 한계.

    cv2.convexHull 은 한 직선 위에 놓인 점들 중 양 끝만 껍질 정점으로 남기고
    중간 점들을 버린다. 손끝 높이가 모두 같으면 손 위쪽 전체가 **껍질 변 하나**가
    되고, convexityDefects 는 그 변 구간에서 가장 깊은 결함 **1개만** 보고한다.
    손가락이 3개든 5개든 골이 1개로 읽히는 이유가 이것이다.

    실제 손은 가운데 손가락이 가장 길어 이 상황이 잘 생기지 않지만, 손을 옆에서
    비스듬히 잡거나 손가락을 가지런히 모으면 근접한 상태가 된다. 실사진 픽스처의
    보(PAPER)가 실패하는 원인 중 하나이기도 하다(README 실패 분석 참조).
    """
    contour = largest_contour(make_synthetic_hand(n_fingers, arc=0))
    gaps = count_finger_gaps(contour, convexity_defects(contour))
    assert gaps == 1, "손끝이 일직선이면 손가락 개수와 무관하게 골이 1개로 뭉개진다"
    assert gaps != n_fingers - 1, "즉 기대값(n-1)과 어긋난다"


def test_둥근_손끝이_일직선_손끝보다_골을_잘_잡는다():
    """arc 파라미터가 실제로 결과를 가르는지 확인 — 위 두 테스트의 대조군."""
    curved = largest_contour(make_synthetic_hand(5, arc=8))
    flat = largest_contour(make_synthetic_hand(5, arc=0))
    n_curved = count_finger_gaps(curved, convexity_defects(curved))
    n_flat = count_finger_gaps(flat, convexity_defects(flat))
    assert n_curved > n_flat


def test_손가락_1개면_골이_0개다():
    """골 0개는 주먹과 손가락 1개를 구분하지 못한다는 사실 자체를 고정한다.

    이 성질이 count_extended_fingers 가 solidity 를 추가로 보는 이유다
    (test_gesture.py 참조). 나중에 누가 solidity 분기를 지우면 이 테스트가
    아니라 저쪽 테스트가 깨지도록 역할을 나눠 두었다.
    """
    contour = largest_contour(make_synthetic_hand(1))
    assert count_finger_gaps(contour, convexity_defects(contour)) == 0


def test_convexity_defects는_N행4열_배열을_반환한다():
    """OpenCV 버전 호환 회귀 테스트.

    OpenCV 4.x 는 (N,1,4), 5.x 는 (N,4) 를 반환한다. 인터넷 예제가 쓰는
    `defects[i, 0]` 인덱싱은 5.x 에서 TypeError 로 죽는다. 우리 래퍼가
    항상 (N,4) 로 정규화하는지 고정한다.
    """
    contour = largest_contour(make_synthetic_hand(5))
    defects = convexity_defects(contour)
    assert defects.ndim == 2
    assert defects.shape[1] == 4
    assert defects.shape[0] > 0


def test_컨투어가_너무_작으면_빈_결함배열을_반환한다():
    """None 이 아니라 (0,4) 빈 배열이어야 호출부가 None 검사를 안 해도 된다."""
    tiny = np.array([[[0, 0]], [[1, 0]], [[1, 1]]], dtype=np.int32)
    defects = convexity_defects(tiny)
    assert defects.shape == (0, 4)


def test_빈_마스크에서는_컨투어를_찾지_못한다():
    assert largest_contour(np.zeros((200, 200), dtype=np.uint8)) is None


def test_최소넓이보다_작은_덩어리는_손으로_보지_않는다():
    """노이즈 얼룩이 '손'으로 승격되면 뒤 단계 전체가 무의미해진다."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (30, 30), 255, -1)   # 넓이 약 441px
    assert largest_contour(mask, min_area=3000) is None
    assert largest_contour(mask, min_area=100) is not None


def test_깊이_임계값이_얕은_골을_거른다():
    """깊이 필터가 실제로 작동하는지 — 임계값을 올리면 골이 사라져야 한다."""
    contour = largest_contour(make_synthetic_hand(5))
    defects = convexity_defects(contour)
    assert count_finger_gaps(contour, defects, min_depth=10.0) == 4
    # 손가락 길이(150px)보다 깊은 골은 존재할 수 없다.
    assert count_finger_gaps(contour, defects, min_depth=500.0) == 0


def test_각도_임계값이_둔각_골을_거른다():
    """각도 필터 작동 확인 — 0도까지 조이면 어떤 골도 통과하지 못한다."""
    contour = largest_contour(make_synthetic_hand(5))
    defects = convexity_defects(contour)
    assert count_finger_gaps(contour, defects, max_angle_deg=90.0) == 4
    assert count_finger_gaps(contour, defects, max_angle_deg=0.0) == 0
