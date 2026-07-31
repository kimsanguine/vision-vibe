"""MediaPipe HandLandmarker 래퍼 — VIDEO 모드 기반 경량 손 추적 엔진.

이 모듈은 프레임을 '받아서' 랜드마크를 돌려주기만 한다. 카메라는 열지 않는다
(카메라 소유는 app.py). 반환은 항상 hand_lite.types 의 계약 타입이며,
MediaPipe 원시 객체는 이 모듈 밖으로 나가지 않는다.

IMAGE 모드가 아니라 VIDEO 모드를 쓰는 이유:
    IMAGE 모드는 매 프레임을 독립 이미지로 취급해 팜 디텍터(손 위치를 찾는
    무거운 쪽 모델)를 매 프레임 다시 돌린다. VIDEO 모드는 직전 프레임의
    랜드마크로 다음 프레임의 손 위치를 예측하고, 추적이 유지되는 동안
    팜 디텍터 호출을 건너뛴다. '경량'이 이 프로젝트의 존재 이유이므로
    이 차이가 핵심이다.
    대신 VIDEO 모드에는 timestamp_ms 가 **단조 증가**해야 한다는 제약이 붙는다.

사용 예:
    with HandLite(num_hands=2) as engine:
        hands = engine.detect(frame_bgr, timestamp_ms=frame_index * 33)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from hand_lite.types import HandResult, Landmark

# models/ 는 패키지(hand_lite/)의 형제 디렉토리다. __file__ 기준으로 절대경로화해
# 어느 작업 디렉토리에서 실행하든 같은 모델을 가리키게 한다.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"

_MODEL_DOWNLOAD_HINT = (
    "다음 명령으로 1회 다운로드하세요:\n"
    '    curl -L -o models/hand_landmarker.task \\\n'
    '      "https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
    'hand_landmarker/float16/1/hand_landmarker.task"'
)


class HandLite:
    """손 랜드마크 추출 엔진. 프레임 1장 → HandResult 리스트.

    VIDEO 모드로 동작하므로 detect() 에 넘기는 timestamp_ms 는 호출할 때마다
    이전 값보다 커야 한다(같아도 안 된다). 프레임 인덱스 기반 타임스탬프
    (frame_index * (1000 // fps))가 가장 안전하다 — 벽시계(time.time())는
    빠른 루프에서 같은 밀리초가 두 번 나올 수 있다.
    """

    def __init__(self, model_path: str | None = None, num_hands: int = 2) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not self.model_path.is_file():
            # 조용한 폴백 금지 — 모델이 없으면 여기서 즉시 끝낸다.
            # (MediaPipe 도 FileNotFoundError 를 내지만 '어떻게 고치는지'는 알려주지 않는다.)
            raise FileNotFoundError(
                f"모델 파일이 없습니다: {self.model_path}\n{_MODEL_DOWNLOAD_HINT}"
            )

        self.num_hands = num_hands
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,  # 핵심: 프레임 간 트래킹 재사용
            num_hands=num_hands,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms: int | None = None

    def detect(self, frame_bgr, timestamp_ms: int) -> list[HandResult]:
        """BGR 프레임에서 손을 검출한다. 손이 없으면 빈 리스트.

        frame_bgr: OpenCV BGR ndarray (H, W, 3).
        timestamp_ms: 직전 호출보다 반드시 큰 정수.
        """
        if self._landmarker is None:
            raise RuntimeError(
                "이미 close() 된 HandLite 입니다. 새 인스턴스를 만드세요."
            )

        # 타임스탬프 역행/정체 처리 방침: 조용히 보정하지 않고 즉시 예외를 던진다.
        # 이유 (1) MediaPipe 자신이 같은 조건에서 ValueError 를 던진다. 여기서
        #   last+1 로 몰래 보정하면 라이브러리 계약과 어긋나는 동작이 생긴다.
        # 이유 (2) 역행/정체는 대개 호출자의 시계 버그(타임스탬프 상수 전달,
        #   루프 재시작 시 카운터 미초기화)다. 보정해주면 코드는 '돌아가지만'
        #   VIDEO 모드가 엉터리 시간축 위에서 추적하게 되어 품질 저하가
        #   원인 불명으로 남는다. 시끄럽게 실패하는 쪽이 디버깅 가능하다.
        # 검증 메시지에 실제 값을 넣어 호출자가 바로 원인을 보게 한다.
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise ValueError(
                f"timestamp_ms 는 단조 증가해야 합니다(VIDEO 모드 제약). "
                f"직전={self._last_timestamp_ms}, 이번={timestamp_ms}. "
                f"프레임 인덱스 기반 타임스탬프 사용을 권장합니다."
            )

        if frame_bgr is None:
            # cap.read() 실패 시 (False, None) 이 흔하다. cv2 내부 에러보다
            # 이 메시지가 원인을 빨리 알려준다.
            raise ValueError("frame_bgr 이 None 입니다. 프레임 읽기가 실패했는지 확인하세요.")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"BGR 3채널 프레임이 필요합니다. 받은 shape={frame_bgr.shape}")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # mp.Image 는 연속 메모리를 요구한다. cvtColor 출력은 연속이지만
        # 슬라이스된 프레임이 들어오는 경우를 대비해 보장한다.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        self._last_timestamp_ms = timestamp_ms
        return self._to_hand_results(result)

    @staticmethod
    def _to_hand_results(result) -> list[HandResult]:
        """MediaPipe 원시 결과 → 계약 타입(HandResult) 변환.

        score 는 MediaPipe 가 결과에 노출하는 유일한 신뢰도인
        handedness 분류 확률이다(Tasks API 는 별도의 손 검출 점수를 주지 않는다).
        """
        hands: list[HandResult] = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            category = handedness[0]
            hands.append(
                HandResult(
                    landmarks=[Landmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks],
                    handedness=category.category_name,
                    score=float(category.score),
                )
            )
        return hands

    def close(self) -> None:
        """네이티브 리소스 해제. 여러 번 불러도 안전하다."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> HandLite:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
