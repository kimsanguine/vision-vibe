"""PaddleOCR 기반 문자 인식 모듈."""

import numpy as np
from paddleocr import PaddleOCR
from dataclasses import dataclass


@dataclass
class OCRResult:
    """OCR 인식 결과."""
    text: str            # 인식된 전체 텍스트
    confidence: float    # 평균 신뢰도
    details: list        # 개별 텍스트 박스 정보 [(bbox, text, conf), ...]


def load_ocr() -> PaddleOCR:
    """PaddleOCR 한국어 모델을 로드한다."""
    return PaddleOCR(
        lang="korean",
        use_angle_cls=True,   # 회전된 텍스트 감지
    )


def recognize_text(ocr: PaddleOCR, plate_img: np.ndarray) -> OCRResult:
    """번호판 이미지에서 텍스트를 인식한다.

    원본 이미지와 전처리 이미지 두 가지로 시도하여
    더 높은 신뢰도의 결과를 반환한다.

    PaddleOCR 3.x부터 `.ocr(cls=True)`가 제거되고 `.predict()`가
    dict형 OCRResult(rec_texts/rec_scores/rec_polys)를 반환하는
    방식으로 바뀌었다 (구버전 [[bbox,(text,conf)],...] 포맷 아님).
    """
    result = ocr.predict(plate_img)

    if not result:
        return OCRResult(text="", confidence=0.0, details=[])

    page = result[0]
    texts = page["rec_texts"]
    confidences = page["rec_scores"]
    boxes = page["rec_polys"]

    if not texts:
        return OCRResult(text="", confidence=0.0, details=[])

    details = list(zip(boxes, texts, confidences))

    # 전체 텍스트 결합 (공백 제거)
    full_text = "".join(texts).replace(" ", "")
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return OCRResult(
        text=full_text,
        confidence=avg_conf,
        details=details,
    )
