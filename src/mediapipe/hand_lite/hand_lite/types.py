"""모듈 간 공유 계약 — 병렬 작업의 인터페이스 고정용.

이 파일은 계약이다. landmarker.py / gesture.py / hud.py 는 모두 이 타입만
주고받는다. 각 모듈은 서로의 내부 구현을 몰라도 되고, 이 파일을 수정하지 않는다.

좌표계 규약(중요):
- Landmark.x, y 는 **정규화 좌표**(0.0~1.0). 픽셀이 아니다.
  화면에 그릴 때만 int(x * width), int(y * height) 로 환산한다.
- z 는 손목(landmark 0) 기준 상대 깊이. 값이 작을수록 카메라에 가깝다.
- MediaPipe HandLandmarker 21점 인덱스 규약을 그대로 따른다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# MediaPipe Hand Landmark 21점 인덱스 (손목 0, 각 손가락 끝 4/8/12/16/20)
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# 관절 연결선 — 랜드마크를 뼈대로 그릴 때 사용
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),          # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),     # 중지
    (9, 13), (13, 14), (14, 15), (15, 16),   # 약지
    (13, 17), (17, 18), (18, 19), (19, 20),  # 소지
    (0, 17),                                  # 손바닥 아래쪽
)


@dataclass(frozen=True)
class Landmark:
    """손 랜드마크 1점. x, y 는 정규화 좌표(0~1)."""

    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class HandResult:
    """검출된 손 1개.

    landmarks: 항상 21개. 인덱스는 MediaPipe 규약.
    handedness: 'Left' 또는 'Right' (카메라 기준이 아니라 실제 손 기준).
    score: 손 검출 신뢰도 0~1.
    """

    landmarks: list[Landmark]
    handedness: str
    score: float


class Gesture(Enum):
    """가위바위보 + 판정 불가."""

    ROCK = "바위"
    PAPER = "보"
    SCISSORS = "가위"
    UNKNOWN = "판정불가"
