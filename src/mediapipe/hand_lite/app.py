"""경량 MediaPipe 손인식 — 실행 진입점.

실행:
    ../../../venv/bin/python app.py                            # 웹캠 모드
    ../../../venv/bin/python app.py --image photo.jpg           # 정지 이미지 모드

정지 이미지 모드가 핵심인 이유:
    이 저장소를 다루는 개발/CI 환경 다수는 웹캠도 디스플레이도 없다.
    --image 는 카메라 없이 로드→추론→제스처 판정→렌더 전체 파이프라인을
    검증할 수 있는 유일한 경로다. 화면에 띄우지 않고 --out 디렉터리에
    결과 이미지를 저장한다(기본값: output/).

조작(웹캠 모드): [Q] 종료
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "hand_landmarker.task"
WINDOW = "MediaPipe Hand Lite"

# VIDEO 모드 타임스탬프를 실측 시간(time.monotonic()) 대신 프레임 인덱스로 만드는 이유:
# hand_lite/landmarker.py 의 주석이 명시적으로 이 방식을 권장한다 — 빠른 루프에서는
# 벽시계 기반 ms가 같은 값을 두 번 낼 수 있어 VIDEO 모드의 "단조 증가" 제약을 어길
# 위험이 있다. 30fps 가정(33ms 간격)의 프레임 카운터는 항상 엄격히 증가한다.
FRAME_INTERVAL_MS = 33


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="경량 MediaPipe 손인식 데모 (웹캠 / 정지 이미지)")
    p.add_argument("--image", default=None,
                   help="정지 이미지 경로. 지정하면 웹캠 대신 이 이미지 1장을 처리하고 파일로 저장한다")
    p.add_argument("--out", default="output", help="이미지 모드 결과 저장 디렉터리. 기본값: output/")
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="HandLandmarker .task 모델 경로")
    p.add_argument("--num-hands", type=int, default=2, help="동시 추적할 최대 손 개수. 기본값: 2")
    return p.parse_args()


def open_capture() -> cv2.VideoCapture:
    """카메라 인덱스 0, 1, 2 순서로 시도해 열리는 첫 캡처를 반환한다."""
    for idx in range(3):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"[카메라] 인덱스 {idx}번 사용")
            return cap
        cap.release()
    sys.exit(
        "웹캠을 열 수 없습니다. 다른 앱이 카메라를 점유 중인지, macOS라면\n"
        "  시스템 설정 > 개인정보 보호 및 보안 > 카메라 권한을 확인하세요."
    )


def run_image_mode(args: argparse.Namespace) -> None:
    """카메라 없이 파이프라인(로드→추론→판정→렌더)을 정지 이미지 1장으로 검증한다."""
    # landmarker/gesture 는 여기서 지연 임포트한다 — 두 모듈이 아직 없어도
    # `app.py --help` 같은 웹캠 모드 무관 경로는 항상 동작해야 하기 때문.
    from hand_lite.gesture import classify
    from hand_lite.hud import draw_hand, draw_label, draw_stats
    from hand_lite.landmarker import HandLite

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"이미지 파일이 없습니다: {image_path}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        sys.exit(f"이미지를 읽지 못했습니다(지원 형식인지 확인하세요): {image_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_result.png"

    with HandLite(model_path=args.model, num_hands=args.num_hands) as landmarker:
        hands = landmarker.detect(frame, timestamp_ms=0)  # 정지 이미지 1장 → 타임스탬프 0 고정

    print(f"[검출] 손 {len(hands)}개")
    for i, hand in enumerate(hands):
        gesture = classify(hand)
        print(f"  손 {i + 1}: {hand.handedness}, {gesture.value} (score={hand.score:.2f})")
        draw_hand(frame, hand)
        draw_label(frame, hand, gesture)
    draw_stats(frame, fps=0.0, hand_count=len(hands))

    cv2.imwrite(str(out_path), frame)
    print(f"[완료] 결과 저장: {out_path}")


def run_webcam_mode(args: argparse.Namespace) -> None:
    from hand_lite.gesture import classify
    from hand_lite.hud import draw_hand, draw_label, draw_stats
    from hand_lite.landmarker import HandLite

    cap = open_capture()
    landmarker = HandLite(model_path=args.model, num_hands=args.num_hands)

    print("조작: [Q] 종료\n")

    frame_index = 0
    prev_tick = time.monotonic()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("카메라 프레임을 읽지 못했습니다. 종료합니다.")
                break

            frame = cv2.flip(frame, 1)  # 거울 모드

            timestamp_ms = frame_index * FRAME_INTERVAL_MS
            hands = landmarker.detect(frame, timestamp_ms)
            frame_index += 1

            for hand in hands:
                gesture = classify(hand)
                draw_hand(frame, hand)
                draw_label(frame, hand, gesture)

            now = time.monotonic()
            elapsed = now - prev_tick
            prev_tick = now
            if elapsed > 0:
                fps = 1.0 / elapsed
            draw_stats(frame, fps, len(hands))

            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


def main() -> None:
    args = parse_args()
    if args.image:
        run_image_mode(args)
    else:
        run_webcam_mode(args)


if __name__ == "__main__":
    main()
