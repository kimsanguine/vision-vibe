"""전체 파이프라인 조립 — BGR 프레임 하나를 받아 손모양 하나를 낸다.

    프레임 → 피부 마스크(skin) → 최대 컨투어(fingers) → 골 개수 →
    손가락 개수 → 손모양(gesture)

중간 산출물(HandShape, 마스크, 컨투어)을 함께 반환하는 이유: 이 프로젝트의
산출물은 "맞췄다/틀렸다" 숫자가 아니라 **어디서 무너졌는지**다. 최종 Gesture
만 돌려주면 마스크 실패와 기하 실패를 구분할 수 없다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fingers import analyze_contour, largest_contour
from .gesture import classify
from .skin import skin_mask
from .types import Gesture, HandShape


@dataclass(frozen=True)
class Detection:
    """프레임 1장의 판정 결과 + 진단 정보.

    shape 가 None 이면 "피부색 덩어리를 찾지 못했다"(마스크 단계 실패),
    shape 가 있는데 gesture 가 UNKNOWN 이면 "손은 찾았는데 손가락 수가
    애매했다"(기하 단계 실패)로 원인이 갈린다.
    """

    gesture: Gesture
    shape: HandShape | None
    mask: np.ndarray
    contour: np.ndarray | None


def detect(frame_bgr: np.ndarray) -> Detection:
    """프레임 1장 → Detection."""
    mask = skin_mask(frame_bgr)
    contour = largest_contour(mask)
    if contour is None:
        return Detection(gesture=Gesture.UNKNOWN, shape=None, mask=mask, contour=None)

    shape = analyze_contour(contour)
    return Detection(
        gesture=classify(shape.fingers),
        shape=shape,
        mask=mask,
        contour=contour,
    )
