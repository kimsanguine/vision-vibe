"""제스처 판정 레이어 — 순수 함수, 결정적 기하 판정.

MediaPipe도 OpenCV도 카메라도 import하지 않는다. 입력은 계약 타입
`HandResult` 하나, 출력은 `Gesture` 하나뿐이다. "손가락이 펴졌는가"는
좌표 두 개를 비교하면 답이 나오는 문제이므로, 여기에 학습된 분류기를
끼워넣을 이유가 없다 — if문으로 충분한 곳에 모델을 쓰지 않는다.

좌표계는 types.py 규약을 그대로 따른다: x, y는 정규화 좌표(0~1),
화면 위쪽일수록 y가 작다.

--- 엄지 규칙 변경 이력 (실사진 검증 결과 반영) ---
최초 구현은 엄지를 "x축 tip vs IP 비교 + handedness로 부등호 반전"으로
판정했다. `../../yolo/rps/tests/fixtures/`의 실제 손 사진 6장으로 검증한 결과
검지~소지 규칙은 6/6 정확했지만 엄지는 0/6 — 원인 두 가지가 겹쳤다:
① 사진이 전부 손등이 보이는 각도였고, ② 주먹을 쥐면 엄지가 접힌 손가락
위에 얹히면서 x좌표만으로는 "펴짐"으로 잘못 읽혔다. 지금은 "엄지 끝이
검지 밑동(INDEX_MCP)에서 얼마나 멀리 떨어졌는가"를 손 크기로 정규화한
거리비로 판정한다 — x/y축 방향에 의존하지 않으므로 handedness가 필요
없어졌고, 그 결과 app.py의 cv2.flip(거울 모드) 여부와도 무관해졌다.
"""

from __future__ import annotations

import math

from .types import Gesture, HandResult

# 검지~소지: (MCP, PIP, DIP, TIP) 인덱스. TIP과 비교할 기준 관절은 PIP.
_INDEX = (5, 6, 7, 8)
_MIDDLE = (9, 10, 11, 12)
_RING = (13, 14, 15, 16)
_PINKY = (17, 18, 19, 20)

_WRIST = 0
_THUMB_TIP = 4
_INDEX_MCP = 5
_MIDDLE_MCP = 9

# 엄지-검지밑동 거리를 "손목-중지밑동 거리"(손 크기 기준)로 나눈 비율.
# 실사진 6장(../../yolo/rps/tests/fixtures/) 실측값:
#   굽음(rock)  0.217, 0.266                    → 최대 0.266
#   펴짐(paper·scissors) 0.865, 0.900, 0.923, 1.058 → 최소 0.865
# 두 무리 사이가 0.266~0.865로 폭 0.598만큼 비어 있다. 그 구간의 가장자리가
# 아니라 **중앙**을 잡아야 새 표본이 들어와도 오판 여지가 가장 적다.
#   0.40 → 마진 0.134 / 0.75 → 마진 0.115 / 0.57 → 마진 0.295
# 셋 다 이 6장은 6/6으로 맞히지만, 중앙값만 마진이 2배 이상이다.
# 표본이 6장뿐이므로 손 크기·카메라 각도가 크게 다른 사용자에게는
# 재보정이 필요할 수 있다.
_THUMB_SPREAD_RATIO_THRESHOLD = 0.57


def fingers_up(hand: HandResult) -> tuple[bool, bool, bool, bool, bool]:
    """엄지, 검지, 중지, 약지, 소지 순으로 펴짐 여부.

    검지~소지 규칙: tip.y < pip.y 면 펴진 것으로 본다 — 손끝이 두 번째
    관절보다 화면 위쪽에 있다는 뜻이다.
    한계: 이 규칙은 손이 세워진 자세(손가락이 대략 위쪽을 향함)를 전제한다.
    손이 옆으로 누우면(예: 손을 수평으로 돌린 경우) tip과 pip의 y 차이가
    작아지거나 부호가 뒤집혀 오판정될 수 있다. 이 함수는 손의 전체 회전을
    보정하지 않는다.

    엄지 규칙: "엄지 끝이 검지 밑동에서 얼마나 벌어졌는가"를 손 크기
    (손목-중지밑동 거리)로 정규화한 비율로 판정한다. x/y 어느 축으로
    뻗었는지 방향을 안 보므로 handedness('Left'/'Right')를 쓰지 않는다
    — 이 함수는 hand.handedness 를 아예 참조하지 않는다.
    한계: 손 크기 정규화 기준(손목-중지밑동)이 카메라 원근에 의해
    왜곡되면(손이 카메라 쪽으로 기울어 짧게 찍히면) 분모가 작아져 비율이
    과대평가되고 엄지가 항상 "펴짐"으로 오판될 수 있다. 또한 이 지표는
    "검지 밑동과의 거리"만 보므로, 엄지가 다른 손가락 사이에 끼는
    제스처(예: OK 사인)처럼 검지 밑동과는 멀지만 실제로는 안 펴진
    모양은 구분하지 못한다.
    """
    if len(hand.landmarks) != 21:
        raise ValueError(f"landmarks must have exactly 21 points, got {len(hand.landmarks)}")

    lm = hand.landmarks

    def _is_extended(finger: tuple[int, int, int, int]) -> bool:
        _mcp, pip, _dip, tip = finger
        return lm[tip].y < lm[pip].y

    index = _is_extended(_INDEX)
    middle = _is_extended(_MIDDLE)
    ring = _is_extended(_RING)
    pinky = _is_extended(_PINKY)

    def _dist(a: int, b: int) -> float:
        return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)

    hand_size = _dist(_WRIST, _MIDDLE_MCP)
    if hand_size == 0:
        # 랜드마크가 전부 한 점에 뭉친 손상된 검출 — 벌어짐을 판정할 기준이
        # 없으므로 안전하게 "안 펴짐"으로 취급한다.
        thumb = False
    else:
        spread_ratio = _dist(_THUMB_TIP, _INDEX_MCP) / hand_size
        thumb = spread_ratio > _THUMB_SPREAD_RATIO_THRESHOLD

    return (thumb, index, middle, ring, pinky)


def classify(hand: HandResult) -> Gesture:
    """가위바위보 판정. 애매하면 Gesture.UNKNOWN.

    주먹(다섯 손가락 전부 굽음)=바위, 다섯 손가락 전부 펴짐=보,
    검지+중지가 펴지고 약지+소지가 굽으면=가위(엄지는 무관). 그 외 조합은
    억지로 분류하지 않고 UNKNOWN.

    엄지를 SCISSORS 조건에서 뺀 이유: 실사진 검증 결과 실제 사람은 가위를
    낼 때 엄지를 벌린 채로 낸다(scissors_0/1.png 둘 다 엄지가 벌어져 있음).
    "검지+중지만" 펴짐을 요구하면 현실의 가위 제스처가 전부 UNKNOWN으로
    빠지므로, 가위 판정에서는 엄지 상태를 보지 않는다.

    랜드마크 개수가 21개가 아닌 손상된 검출 결과(예: 카메라 프레임 드롭)가
    들어오면 fingers_up이 ValueError를 던지는데, 이 함수는 그걸 흡수해
    UNKNOWN을 반환한다 — 실시간 앱 루프가 검출 실패 한 번에 죽지 않게 하기
    위함이다.
    """
    try:
        thumb, index, middle, ring, pinky = fingers_up(hand)
    except ValueError:
        return Gesture.UNKNOWN

    if not any((thumb, index, middle, ring, pinky)):
        return Gesture.ROCK
    if all((thumb, index, middle, ring, pinky)):
        return Gesture.PAPER
    if index and middle and not ring and not pinky:
        return Gesture.SCISSORS
    return Gesture.UNKNOWN
