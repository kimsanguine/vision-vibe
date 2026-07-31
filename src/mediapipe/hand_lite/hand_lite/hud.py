"""HUD 그리기 전담 — MediaPipe import 금지, OpenCV(+PIL)만 사용.

한글 렌더링 문제(중요):
    cv2.putText 는 라틴 문자 전용 내장 폰트만 지원해 한글이 "???"로 깨진다.
    hand_lite.types.Gesture 의 값이 "바위"/"보"/"가위"/"판정불가"로 한국어라
    draw_label()/draw_stats() 는 이 문제를 반드시 만난다.
    해법은 같은 워크스페이스의 yolo/rps/rps/hud.py 와 동일하게 맞춘다:
    PIL ImageDraw 로 텍스트를 그린 뒤 numpy BGR 프레임으로 되돌린다.
    새 의존성은 없다 — Pillow는 mediapipe/opencv-python 설치 시 이미
    함께 들어와 있다(venv에서 `pip show pillow`로 확인함).
    영문 매핑 대신 이 방식을 고른 이유: Gesture.value 자체를 그대로 쓸 수
    있어 gesture.py 와의 이름 매핑 테이블을 따로 유지할 필요가 없다.

좌표계: hand_lite.types 규약대로 랜드마크는 정규화 좌표(0~1)다.
    프레임에 그릴 때는 반드시 int(x * width), int(y * height) 로 환산한다.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hand_lite.types import HAND_CONNECTIONS, Gesture, HandResult

# cv2 그리기용 색상 (BGR)
SKELETON_COLOR = (60, 220, 60)
JOINT_COLOR = (40, 140, 255)

# PIL 텍스트용 색상 (RGB) — cv2 색상과 채널 순서가 다르므로 분리해서 헷갈림 방지
LABEL_COLOR = (255, 255, 255)
STATS_COLOR = (120, 255, 120)

# macOS 시스템 한글 폰트 — yolo/rps/rps/hud.py 와 동일 경로를 우선 시도
_KO_FONT_PATHS = (
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
)

_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _korean_font(size: int) -> ImageFont.FreeTypeFont:
    """크기별 한글 폰트를 로드해 캐시한다. 시스템 폰트가 없으면 기본 폰트로 폴백."""
    if size in _font_cache:
        return _font_cache[size]
    for path in _KO_FONT_PATHS:
        try:
            font = ImageFont.truetype(path, size)
            break
        except OSError:
            continue
    else:
        font = ImageFont.load_default(size)
    _font_cache[size] = font
    return font


def _draw_korean_text(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    size: int = 24,
    color: tuple[int, int, int] = LABEL_COLOR,
) -> None:
    """frame(BGR numpy, in-place 수정) 위에 한글 텍스트를 합성한다.

    PIL 라운드트립(BGR→RGB→PIL→numpy→BGR) 비용이 있지만, 이 프로젝트는
    '경량'이 목표이지 '실시간 초저지연'이 목표가 아니라서 프레임당 몇 번
    (라벨 수만큼 + 통계 1번) 발생해도 감내 가능한 수준이다.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    ImageDraw.Draw(canvas).text(org, text, font=_korean_font(size), fill=color)
    frame[:] = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def draw_hand(frame: np.ndarray, hand: HandResult) -> None:
    """HAND_CONNECTIONS 뼈대 + 관절점을 그린다.

    랜드마크는 정규화 좌표(0~1)이므로 frame 크기로 환산해야 한다.
    """
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], SKELETON_COLOR, 2, cv2.LINE_AA)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, JOINT_COLOR, -1, cv2.LINE_AA)


def draw_label(frame: np.ndarray, hand: HandResult, gesture: Gesture) -> None:
    """손 위쪽(가장 높은 랜드마크 근처)에 제스처 이름을 한글로 표시한다."""
    h, w = frame.shape[:2]
    top_y = min(lm.y for lm in hand.landmarks) * h
    center_x = sum(lm.x for lm in hand.landmarks) / len(hand.landmarks) * w
    text = f"{gesture.value} ({hand.score:.2f})"
    org = (max(0, int(center_x) - 50), max(0, int(top_y) - 40))
    _draw_korean_text(frame, text, org, size=26, color=LABEL_COLOR)


def draw_stats(frame: np.ndarray, fps: float, hand_count: int) -> None:
    """좌상단에 FPS와 검출된 손 개수를 표시한다."""
    text = f"FPS: {fps:.1f}   손 개수: {hand_count}"
    _draw_korean_text(frame, text, (12, 8), size=24, color=STATS_COLOR)


# rps_game.py(가위바위보 모드) 전용 — draw_hand/draw_label/draw_stats 위에
# 점수판 + 라운드 상태만 얹는다. 손 그리기 자체는 이미 위 함수들이 담당하므로
# 중복 구현하지 않는다.
RESULT_COLOR = (255, 255, 255)  # PIL RGB


def draw_scoreboard(frame: np.ndarray, wins: int, losses: int, draws: int) -> None:
    """좌상단에 누적 전적을 표시한다(draw_stats와 같은 자리 대역, 손인식
    데모의 FPS 대신 가위바위보 전적을 보여주는 용도라 위치를 공유한다)."""
    text = f"{wins}승 {losses}패 {draws}무"
    _draw_korean_text(frame, text, (12, 8), size=24, color=STATS_COLOR)


def draw_round_status(frame: np.ndarray, status_text: str) -> None:
    """draw_scoreboard 바로 아래 줄에 현재 라운드 상태(대기/판정 대기/결과)를
    표시한다.

    처음에는 화면 중앙에 띄우려 했으나(텍스트 길이 * 대략적인 글자폭으로
    x좌표 계산), 한글+영문이 섞인 가변폭 텍스트에서 그 추정이 부정확해
    draw_scoreboard와 같은 y좌표(8)에서 겹치는 걸 실제 --image 출력에서
    확인했다. 중앙 정렬 추정 대신 draw_scoreboard 아래(y=40)에 같은
    좌측 정렬로 쌓아 겹침 자체를 구조적으로 없앤다.
    """
    _draw_korean_text(frame, status_text, (12, 40), size=28, color=RESULT_COLOR)
