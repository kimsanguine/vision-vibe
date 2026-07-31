"""1단계 — YCrCb 피부색 검출.

왜 HSV 가 아니라 YCrCb 인가
────────────────────────────
피부색을 색공간에서 임계값으로 자를 때 문제는 "같은 피부인데 조명이 밝으면
값이 통째로 이동한다"는 것이다. YCrCb 는 밝기(Y)와 색차(Cr, Cb)를 **직교
축으로 분리**하도록 설계된 방송용 색공간이다. 그래서 조명이 바뀌면 Y 만
움직이고 Cr/Cb 는 상대적으로 덜 움직인다 — Y 를 아예 임계에서 빼버리면
(아래 LOWER 의 첫 성분이 0, UPPER 가 255 인 이유가 이것이다) 밝기 변화를
설계 수준에서 무시할 수 있다.

HSV 의 Hue 도 비슷한 의도의 축이지만 두 가지가 불리하다.
  1. Hue 는 원형(0도와 179도가 이웃)이라 빨강 근처인 피부색이 배열 양 끝으로
     갈라진다. cv2.inRange 는 선형 구간만 자르므로 범위를 두 번 잘라 OR 해야
     한다 — 경계 처리가 지저분해지고 실수하기 쉽다.
  2. 채도(S)가 낮은 픽셀에서 Hue 는 수치적으로 불안정하다. 밝은 조명에 날아간
     손등 하이라이트가 정확히 그 영역이다.
YCrCb 의 Cr/Cb 는 직교 좌표라 두 문제가 모두 없다.

임계값의 출처
────────────
Cr∈[133,173], Cb∈[77,127] 은 Chai & Ngan, "Face segmentation using
skin-color map in videophone applications" (IEEE TCSVT, 1999) 에서 제시된
고전적 피부색 구간이다. 이 프로젝트가 이 값을 **직접 튜닝해서 얻은 게 아니라
문헌 기본값을 그대로 쓴다**는 점을 명시한다 — 픽스처에 맞춰 값을 흔들면
"이 6장에서만 되는 숫자"가 되어 일반화 주장을 할 수 없게 된다.

한계(측정으로 확인됨, README 참조)
  - 손톱에 매니큐어가 칠해져 있으면 그 부분이 Cr 범위 밖으로 나가 **손끝이
    잘린다**. 손가락이 짧아지면 손가락 사이 골이 얕아져 3단계 검출이 무너진다.
  - 배경에 나무·베이지 벽·황토색 물체가 있으면 손과 함께 마스킹된다.
"""

from __future__ import annotations

import cv2
import numpy as np

# 문헌 기본값. Y 축은 [0,255] 전체를 허용해 밝기를 판정에서 제외한다.
SKIN_LOWER_YCRCB: tuple[int, int, int] = (0, 133, 77)
SKIN_UPPER_YCRCB: tuple[int, int, int] = (255, 173, 127)


def skin_mask(
    frame_bgr: np.ndarray,
    *,
    kernel_size: int = 5,
    open_iterations: int = 2,
    close_iterations: int = 2,
    blur_ksize: int = 5,
) -> np.ndarray:
    """BGR 프레임에서 피부색 이진 마스크(0/255, uint8)를 만든다.

    순서: 블러 → YCrCb 변환 → inRange → open → close

    각 단계의 이유
      블러 : JPEG/센서 노이즈로 튄 단일 픽셀이 임계값을 넘나드는 것을 먼저
             누른다. 마스킹 **전에** 해야 의미가 있다 — 이진화 후에 블러를
             걸면 0/255 사이 중간값이 생겨 다시 임계값이 필요해진다.
      open : 침식→팽창. 배경에 흩뿌려진 작은 흰 점(오검출)을 지운다.
      close: 팽창→침식. 손 내부에 뚫린 작은 검은 구멍(손등 하이라이트, 반지
             등으로 색이 날아간 곳)을 메운다. close 를 빼면 손 안쪽 구멍이
             컨투어를 쪼개서 "가장 큰 컨투어 = 손" 가정이 깨진다.
    open 을 close 보다 먼저 하는 이유: 노이즈 점을 먼저 지우지 않고 close 를
    걸면 팽창 단계에서 그 점들이 서로 붙어 지울 수 없는 덩어리가 된다.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("빈 프레임입니다")

    if blur_ksize >= 3:
        frame_bgr = cv2.GaussianBlur(frame_bgr, (blur_ksize, blur_ksize), 0)

    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(
        ycrcb,
        np.array(SKIN_LOWER_YCRCB, dtype=np.uint8),
        np.array(SKIN_UPPER_YCRCB, dtype=np.uint8),
    )

    # 타원 커널을 쓰는 이유: 사각 커널은 모폴로지 결과에 축 방향 각진 흔적을
    # 남긴다. 손 윤곽은 둥근 곡선이므로 타원 쪽이 원래 형태를 덜 훼손한다.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    return mask
