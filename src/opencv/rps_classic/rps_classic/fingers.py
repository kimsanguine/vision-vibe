"""2·3단계 — 윤곽선/볼록 껍질 추출과 convexity defects 기반 손가락 개수 세기.

기하 아이디어
─────────────
손을 실루엣으로 보면, 편 손가락들의 **끝점**은 볼록 껍질(convex hull) 위에
놓이고, 손가락 **사이의 골**은 껍질 안쪽으로 깊이 파여 있다. OpenCV 의
convexityDefects 는 이 "껍질과 실제 윤곽 사이의 오목한 구간"을 각각
(start, end, far, depth) 로 돌려준다. far 가 골의 가장 깊은 점이다.

  손가락 N개를 펴면 손가락 사이 골은 N-1개  →  fingers = gaps + 1

문제는 손목 안쪽·주먹의 마디처럼 손가락과 무관한 오목점도 함께 나온다는 것이다.
두 가지로 거른다.

  (a) 깊이 필터 : 얕은 골은 윤곽선의 톱니(계단 현상)일 뿐이다.
  (b) 각도 필터 : far 지점에서 두 손가락 끝을 바라본 사잇각을 코사인 법칙으로
                  구해 90도 이하만 인정한다. 펼친 손가락 사이는 뾰족하게(작은
                  각) 모이지만, 손목이나 손날의 완만한 굴곡은 둔각이다.

각도 계산 — 코사인 법칙
───────────────────────
    a = |start - end|   (두 껍질 접점 사이 거리, far 의 맞은편 변)
    b = |start - far|
    c = |end   - far|
    cos(θ) = (b² + c² - a²) / (2bc),   θ = far 에서의 사잇각

이 함수(defect_angle_deg)는 cv2 를 쓰지 않는 **순수 기하 함수**다. 그래서
알려진 좌표(예: 직각삼각형)를 넣어 정확한 값이 나오는지 합성 데이터로 단위
테스트할 수 있다 — 이 파일에서 유일하게 "정답이 수학적으로 확정되는" 부분이라
반드시 테스트로 고정한다.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .types import HandShape

# ── 튜닝 파라미터 ────────────────────────────────────────────────────────
# 이 값들은 문헌 기본값이 아니라 이 구현의 설계 선택이다. README 의 정확도
# 수치는 아래 기본값 그대로 측정한 것이고, evaluate.py 가 임계값을 훑어
# "다른 값이면 맞았을까"를 별도로 측정한다.

MIN_CONTOUR_AREA = 3000       # 이보다 작은 덩어리는 손으로 보지 않는다(픽셀).
MIN_DEFECT_DEPTH = 10.0       # 골 깊이 하한(픽셀). 윤곽 톱니 제거용.
MAX_DEFECT_ANGLE_DEG = 90.0   # 손가락 사이로 인정할 사잇각 상한(도).

# 0개 골일 때 주먹인지 손가락 1개인지 가르는 solidity 임계값.
# 근거는 아래 count_extended_fingers 의 docstring 참조.
FIST_SOLIDITY = 0.80


def defect_angle_deg(
    start: tuple[float, float],
    end: tuple[float, float],
    far: tuple[float, float],
) -> float:
    """far 지점에서 start·end 를 바라본 사잇각(도)을 코사인 법칙으로 구한다.

    반환 범위는 [0, 180]. far 가 start 또는 end 와 같은 점이면 각이 정의되지
    않으므로 180.0 을 반환한다 — 0.0 이 아니라 180.0 인 이유: 이 함수의 결과는
    "90도 이하면 손가락 사이로 인정"에 쓰인다. 정의되지 않는 입력을 0도로
    돌려주면 **가장 뾰족한 골**로 오인되어 손가락 수를 부풀린다. 판정 불가는
    거절 쪽(둔각)으로 보내는 것이 안전하다.
    """
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    fx, fy = float(far[0]), float(far[1])

    a = math.hypot(sx - ex, sy - ey)   # far 의 맞은편 변
    b = math.hypot(sx - fx, sy - fy)
    c = math.hypot(ex - fx, ey - fy)

    if b == 0.0 or c == 0.0:
        return 180.0

    # 부동소수점 오차로 |cos| 가 1을 아주 살짝 넘으면 acos 가 NaN 을 낸다.
    cos_theta = max(-1.0, min(1.0, (b * b + c * c - a * a) / (2 * b * c)))
    return math.degrees(math.acos(cos_theta))


def largest_contour(mask: np.ndarray, *, min_area: float = MIN_CONTOUR_AREA):
    """마스크에서 넓이가 가장 큰 외곽 컨투어를 반환한다. 없으면 None.

    RETR_EXTERNAL 을 쓰는 이유: 손 안쪽 구멍의 내부 윤곽은 필요 없다.
    CHAIN_APPROX_SIMPLE 대신 **CHAIN_APPROX_NONE** 을 쓴다 —
    convexityDefects 는 껍질 접점 사이의 실제 윤곽 점들을 훑어 가장 깊은
    far 를 찾으므로, 직선 구간을 두 점으로 압축해버리면 골 깊이가 과소평가된다.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None
    return contour


def convexity_defects(contour: np.ndarray) -> np.ndarray:
    """(N, 4) 배열 [start_idx, end_idx, far_idx, depth_x256] 을 반환한다.

    OpenCV 버전 호환(중요): OpenCV 4.x 는 (N, 1, 4) 를, 이 환경의 OpenCV 5.0 은
    (N, 4) 를 반환한다. 인터넷 예제 대부분이 쓰는 `defects[i, 0]` 인덱싱은
    5.0 에서 TypeError 로 죽는다. reshape(-1, 4) 로 두 형태를 모두 흡수한다.

    껍질이 3점 미만이거나 결함이 없으면 (0, 4) 빈 배열을 반환한다 — None 을
    반환하면 호출부마다 None 검사를 따로 해야 해서 조용한 버그가 생긴다.
    """
    empty = np.empty((0, 4), dtype=np.int32)
    if contour is None or len(contour) < 4:
        return empty

    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return empty

    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        # 껍질 인덱스가 단조롭지 않을 때 OpenCV 가 예외를 던진다. 이 경우
        # 결함 정보를 얻을 수 없다는 사실 자체가 답이므로 빈 배열로 처리한다.
        return empty
    if defects is None:
        return empty
    return defects.reshape(-1, 4)


def count_finger_gaps(
    contour: np.ndarray,
    defects: np.ndarray,
    *,
    min_depth: float = MIN_DEFECT_DEPTH,
    max_angle_deg: float = MAX_DEFECT_ANGLE_DEG,
) -> int:
    """깊이·각도 필터를 통과한 결함(=손가락 사이 골) 개수.

    depth 는 OpenCV 가 픽셀 거리에 256을 곱한 고정소수점 정수로 준다.
    256으로 나눠야 픽셀 단위가 된다 — 이 나눗셈을 빠뜨리면 임계값이 실질적으로
    무한대가 되어 모든 결함이 통과한다(흔한 실수).
    """
    gaps = 0
    for s_idx, e_idx, f_idx, depth_fixed in defects:
        if depth_fixed / 256.0 < min_depth:
            continue
        start = contour[s_idx][0]
        end = contour[e_idx][0]
        far = contour[f_idx][0]
        if defect_angle_deg(start, end, far) <= max_angle_deg:
            gaps += 1
    return gaps


def count_extended_fingers(gaps: int, solidity: float, *, fist_solidity: float = FIST_SOLIDITY) -> int:
    """골 개수 + 실루엣 충실도(solidity)로 편 손가락 개수를 추정한다.

    왜 단순히 gaps + 1 이 아닌가 — 0골의 근본적 모호성
    ────────────────────────────────────────────────────
    "손가락 N개 → 골 N-1개" 는 N ≥ 2 에서만 성립한다. N=0(주먹)과 N=1(검지만
    폄)은 **둘 다 골이 0개**라, 결함 정보만으로는 원리적으로 구분되지 않는다.
    그런데 과제 명세는 0개=바위를 요구하므로, gaps+1 을 그대로 쓰면 주먹이
    영원히 1개로 읽혀 바위가 검출 불가능해진다. 즉 0골 구간에는 결함 이외의
    단서가 반드시 하나 필요하다.

    그 단서로 solidity(컨투어 넓이 / 볼록 껍질 넓이)를 쓴다. 주먹은 껍질을
    거의 꽉 채우고(1에 가까움), 손가락 하나가 튀어나오면 껍질에 빈 삼각형이
    생겨 값이 떨어진다. 픽스처 실측: 주먹 0.895 / 보 0.735 / 가위 0.699.
    임계값 0.80 은 그 사이를 가른다.

    [추정] 0.80 은 이 6장에서 관측된 분리 지점이다. 다른 손·각도에서
    같은 값이 유효하다는 근거는 없다. 이 임계값 의존성 자체가 전통 기법의
    한계를 보여주는 사례로 README 에 남겼다.
    """
    if gaps > 0:
        return gaps + 1
    return 0 if solidity >= fist_solidity else 1


def analyze_contour(contour: np.ndarray) -> HandShape:
    """컨투어 하나에서 HandShape(넓이·solidity·골 수·손가락 수)를 계산한다."""
    area = float(cv2.contourArea(contour))
    hull_points = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull_points))
    solidity = area / hull_area if hull_area > 0 else 0.0

    gaps = count_finger_gaps(contour, convexity_defects(contour))
    return HandShape(
        contour_area=area,
        hull_area=hull_area,
        solidity=solidity,
        finger_gaps=gaps,
        fingers=count_extended_fingers(gaps, solidity),
    )
