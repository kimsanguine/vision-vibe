"""번호판 이미지 전처리 모듈."""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class PreprocessResult:
    """전처리 단계별 결과 (시각화용)."""
    original: np.ndarray     # 원본 크롭
    resized: np.ndarray      # 리사이즈
    gray: np.ndarray         # 그레이스케일
    binary: np.ndarray       # 이진화
    denoised: np.ndarray     # 노이즈 제거 (최종)


def preprocess_plate(plate_img: np.ndarray, target_width: int = 520) -> PreprocessResult:
    """번호판 크롭 이미지를 OCR에 적합하도록 전처리한다.

    단계: 리사이즈 → 그레이스케일 → 적응형 이진화 → 노이즈 제거
    """
    original = plate_img.copy()

    # 1. 리사이즈 (가로 기준, 비율 유지)
    h, w = plate_img.shape[:2]
    if w > 0:
        scale = target_width / w
        target_height = int(h * scale)
        resized = cv2.resize(plate_img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    else:
        resized = plate_img.copy()

    # 2. 그레이스케일
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized.copy()

    # 3. CLAHE (적응형 히스토그램 균등화) — 조명 편차 보정
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4. 적응형 이진화
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=4,
    )

    # 5. 노이즈 제거 (모폴로지 연산)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    denoised = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    denoised = cv2.medianBlur(denoised, 3)

    return PreprocessResult(
        original=original,
        resized=resized,
        gray=gray,
        binary=binary,
        denoised=denoised,
    )
