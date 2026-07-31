"""YOLO11 기반 번호판 검출 모듈."""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "license_plate_yolo11n.pt")


@dataclass
class DetectionResult:
    """번호판 검출 결과."""
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    cropped: np.ndarray  # 크롭된 번호판 이미지
    method: str  # "yolo" 또는 "opencv_fallback" — 어떤 경로로 검출됐는지


def load_detector(model_path: str = DEFAULT_MODEL_PATH) -> YOLO:
    """번호판 전용 YOLO11 모델을 로드한다 (출처: Gholamreza/yolo11_license_plate_detection, Apache-2.0)."""
    return YOLO(model_path)


def detect_plates(model: YOLO, image: np.ndarray, conf_threshold: float = 0.25) -> list[DetectionResult]:
    """이미지에서 번호판을 검출한다.

    번호판 전용 YOLO 모델로 먼저 시도하고, 검출 실패 시에만
    OpenCV 기반 검출을 폴백으로 사용한다. 어느 경로로 검출됐는지는
    DetectionResult.method에 남겨 UI에서 그대로 표시한다.
    """
    results = model(image, verbose=False, conf=conf_threshold)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cropped = image[y1:y2, x1:x2]
            if cropped.size > 0:
                detections.append(DetectionResult(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    cropped=cropped,
                    method="yolo",
                ))

    if not detections:
        detections = _detect_plates_opencv(image)

    return detections


def _detect_plates_opencv(image: np.ndarray) -> list[DetectionResult]:
    """OpenCV 기반 번호판 영역 검출 (폴백).

    번호판의 가로:세로 비율(약 2:1~5:1)과
    에지 밀도를 기반으로 후보 영역을 찾는다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape[:2]

    # 양방향 필터로 노이즈 제거 (에지는 보존)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    # 에지 검출
    edges = cv2.Canny(filtered, 30, 200)

    # 팽창으로 에지 연결
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # 컨투어 찾기
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h > 0 else 0
        area_ratio = (w * h) / (w_img * h_img)

        # 번호판 비율 필터: 가로/세로 2~6, 면적 비율 0.5%~15%
        if 2.0 <= aspect_ratio <= 6.0 and 0.005 <= area_ratio <= 0.15:
            # 약간의 마진 추가
            margin_x = int(w * 0.05)
            margin_y = int(h * 0.1)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w_img, x + w + margin_x)
            y2 = min(h_img, y + h + margin_y)

            cropped = image[y1:y2, x1:x2]
            if cropped.size > 0:
                detections.append(DetectionResult(
                    bbox=(x1, y1, x2, y2),
                    confidence=0.5,  # OpenCV 폴백이므로 고정 신뢰도
                    cropped=cropped,
                    method="opencv_fallback",
                ))

    return detections[:3]  # 상위 3개까지
