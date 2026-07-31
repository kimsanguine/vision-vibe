"""모듈 간 공유 계약 — 손모양·승부 결과 enum.

이 프로젝트는 mediapipe/hand_lite / yolo/rps 와 **같은 문제**를 다른 기법으로
푼다. 따라서 손모양 enum 의 의미(가위/바위/보/판정불가)를 그대로 맞춘다.
3자 비교표에서 같은 이름이 같은 뜻이어야 비교가 성립하기 때문이다.

이 파일은 cv2 / numpy 를 import 하지 않는다 — 순수 계약이라 어떤 백엔드에서도
읽을 수 있어야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Gesture(Enum):
    """가위바위보 + 판정 불가.

    UNKNOWN 은 "손이 없다"와 "손은 있는데 셋 중 하나로 확신할 수 없다"를 모두
    포함한다. 둘을 구분하지 않는 이유: 게임 로직 입장에서 두 경우의 처리가
    동일하기 때문이다(라운드를 판정하지 않는다). 진단이 필요할 때는
    HandShape 를 함께 보면 구분된다(None 이면 손 없음).
    """

    ROCK = "바위"
    PAPER = "보"
    SCISSORS = "가위"
    UNKNOWN = "판정불가"


class Result(Enum):
    """플레이어 관점의 라운드 결과."""

    WIN = "승리"
    LOSE = "패배"
    DRAW = "무승부"
    UNDECIDABLE = "판정불가"


@dataclass(frozen=True)
class HandShape:
    """피부색 마스크에서 뽑아낸 손 1개의 기하 요약.

    파이프라인 중간 산출물을 dataclass 로 고정하는 이유: 이 프로젝트의 교육적
    핵심이 "어디서 틀렸나"를 보여주는 것이라, 최종 Gesture 만 반환하면 실패
    원인을 추적할 수 없다. finger_gaps / solidity 를 함께 내보내야
    "마스크는 잘 잡혔는데 골이 얕았다" 같은 진단이 가능하다.

    contour_area : 손 컨투어 넓이(픽셀).
    hull_area    : 볼록 껍질 넓이(픽셀).
    solidity     : contour_area / hull_area. 1에 가까울수록 '꽉 찬' 모양(주먹).
    finger_gaps  : 손가락 사이 골로 인정된 convexity defect 개수.
    fingers      : 추정 편 손가락 개수.
    """

    contour_area: float
    hull_area: float
    solidity: float
    finger_gaps: int
    fingers: int
