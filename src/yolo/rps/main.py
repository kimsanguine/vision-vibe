"""YOLO 가위바위보 — 웹캠으로 낸 손모양을 인식해 AI와 대결한다.

실행:
    venv/bin/python main.py                 # 웹캠(기본 0번 카메라)
    venv/bin/python main.py --source tests/fixtures/rps_video.avi   # 영상 파일로 데모
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from rps.ai_move import prepare_ai_moves
from rps.detector import MoveStabilizer, RPSDetector, pick_device
from rps.game import Game, Phase
from rps.hud import HUD, MUTED, OUTCOME_COLOR, TEXT, draw_detection_box
from rps.judge import OUTCOME_KOREAN

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "rps_yolo11x_leeyunjai.pt"
ASSETS_DIR = PROJECT_ROOT / "assets" / "ai_moves"
WINDOW = "YOLO Rock-Paper-Scissors vs AI"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO 가위바위보 vs AI")
    p.add_argument("--source", default="0",
                   help="카메라 인덱스(예: 0) 또는 영상 파일 경로. 기본값: 0")
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO 모델 경로")
    p.add_argument("--device", default=None,
                   help="추론 디바이스(mps/cuda/cpu). 미지정 시 자동 선택")
    p.add_argument("--imgsz", type=int, default=640,
                   help="추론 입력 크기. 640 미만은 인식률이 급격히 떨어진다")
    p.add_argument("--conf", type=float, default=0.5, help="검출 신뢰도 임계값")
    p.add_argument("--regen-ai-images", action="store_true",
                   help="캐시를 무시하고 AI 패 이미지를 새로 생성")
    return p.parse_args()


# 카메라 캡처 해상도. 1080p 그대로 받으면 ultralytics의 letterbox 전처리 비용이 커져
# 실측 7.7 FPS까지 떨어진다. 720p로 낮추면 14.5 FPS (측정: Apple Silicon / mps / imgsz=640).
CAPTURE_W, CAPTURE_H = 1280, 720


def open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    else:
        path = Path(source)
        if not path.exists():
            sys.exit(f"영상 파일이 없습니다: {path}")
        cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(
            f"영상 소스를 열지 못했습니다: {source}\n"
            "  - 웹캠이라면 다른 앱이 카메라를 점유 중인지, 그리고 macOS\n"
            "    시스템 설정 > 개인정보 보호 > 카메라 권한을 확인하세요."
        )
    return cap


def status_for(game: Game) -> tuple[str, tuple[int, int, int]]:
    if game.phase is Phase.READY:
        return "SPACE를 눌러 시작", TEXT
    if game.phase is Phase.COUNTDOWN:
        return "준비...", TEXT
    if game.missed:
        return "손모양을 인식하지 못했습니다", MUTED
    assert game.outcome is not None
    return OUTCOME_KOREAN[game.outcome], OUTCOME_COLOR[game.outcome]


def main() -> None:
    args = parse_args()
    is_camera = args.source.isdigit()

    device = pick_device(args.device)
    print(f"[모델] {args.model}\n[디바이스] {device} (imgsz={args.imgsz}, conf={args.conf})")
    detector = RPSDetector(args.model, device=device, imgsz=args.imgsz, conf=args.conf)
    print(f"[클래스] {detector.class_names}")

    ai_provider = prepare_ai_moves(PROJECT_ROOT, ASSETS_DIR, regenerate=args.regen_ai_images)
    if not ai_provider.has_images:
        print("[AI 패] 이모지 폴백 모드로 실행합니다.")

    cap = open_capture(args.source)
    game = Game(ai_provider=ai_provider)
    stabilizer = MoveStabilizer()
    hud = HUD()

    print("\n조작: [SPACE] 라운드 시작   [R] 점수 초기화   [Q/ESC] 종료\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_camera:
                    print("카메라 프레임을 읽지 못했습니다. 종료합니다.")
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 영상 파일은 반복 재생
                continue

            if is_camera:
                frame = cv2.flip(frame, 1)  # 거울 모드

            detection = detector.detect(frame)
            stabilizer.update(detection.move if detection else None)
            stable = stabilizer.stable_move()

            if detection is not None:
                draw_detection_box(frame, detection.box, detection.move, detection.confidence)

            game.update(stable)
            status_text, status_color = status_for(game)
            ai_move = game.ai_move
            canvas = hud.render(
                frame,
                score=game.score.as_tuple(),
                status_text=status_text,
                status_color=status_color,
                countdown=game.countdown_number(),
                live_move=stable,
                user_move=game.user_move,
                ai_move=ai_move,
                ai_image_path=ai_provider.image_path(ai_move) if ai_move else None,
                footer_text="SPACE 시작 · R 점수 초기화 · Q 종료",
            )
            cv2.imshow(WINDOW, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                game.start_round()
            elif key == ord("r"):
                game.reset_score()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        w, l, d = game.score.as_tuple()
        print(f"\n최종 전적: {w}승 {l}패 {d}무")


if __name__ == "__main__":
    main()
