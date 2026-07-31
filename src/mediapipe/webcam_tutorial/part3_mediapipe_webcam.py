"""Part 3. MediaPipe 웹캠 손 감지 도전과제 — 스타터 코드.

목적: YOLO(커스텀 학습 필요) vs MediaPipe(구글 사전학습·온디바이스)의
개발 경험 차이를 직접 체감한다. 이 스크립트는 학습 데이터도, 파인튜닝도
없이 곧바로 손 21개 랜드마크를 실시간으로 추적한다.

사전 준비:
    pip install mediapipe opencv-python
    모델 다운로드(1회):
        curl -L -o models/hand_landmarker.task \
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

주의(중요, macOS/Windows 공통 함정):
- macOS: 시스템 설정 > 개인정보 보호 및 보안 > 카메라에서 터미널(또는 실행 앱)에
  카메라 접근을 허용해야 한다. 허용 전에는 VideoCapture가 조용히 빈 프레임만 반환한다.
- Windows: 카메라 인덱스가 0이 아닐 수 있다(내장캠+외장캠 혼재 시). 아래
  `open_camera()`가 0,1,2를 순서대로 시도한다.
- 이 스크립트는 실제 웹캠 하드웨어가 있어야 동작 확인이 가능하다 —
  샌드박스/서버 환경에서는 카메라 오픈 자체가 실패하는 게 정상이다.
"""

import os
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

# 손가락 관절 연결선 (MediaPipe Hand Landmark 21점 기준)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),        # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),   # 중지
    (9, 13), (13, 14), (14, 15), (15, 16), # 약지
    (13, 17), (17, 18), (18, 19), (19, 20),# 소지
    (0, 17),
]


def open_camera(max_index: int = 3) -> cv2.VideoCapture:
    """0번부터 순서대로 시도해 열리는 첫 카메라를 반환한다.

    Windows에서 내장캠/외장캠이 섞여 있으면 0번이 아닐 수 있어서 필요.
    """
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"카메라 인덱스 {idx}번 사용")
            return cap
        cap.release()
    raise RuntimeError(
        "웹캠을 열 수 없습니다. macOS는 카메라 권한 허용 여부, "
        "Windows는 다른 앱이 카메라를 점유 중인지 확인하세요."
    )


def draw_landmarks(frame, hand_landmarks) -> None:
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def main():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"모델 파일이 없습니다: {MODEL_PATH}\n스크립트 상단 주석의 curl 명령으로 먼저 다운로드하세요.")

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = open_camera()
    print("종료하려면 창을 클릭하고 'q'를 누르세요.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("프레임을 읽지 못했습니다.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            for hand_landmarks in result.hand_landmarks:
                draw_landmarks(frame, hand_landmarks)

            cv2.putText(frame, f"hands: {len(result.hand_landmarks)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("MediaPipe Hand Landmarker (q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
