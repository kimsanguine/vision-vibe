"""경량 MediaPipe 가위바위보 — 실행 진입점.

app.py(손인식 데모)와 같은 구조를 그대로 따른다: 웹캠 없이도 카메라·디스플레이가
없는 환경(CI 등)에서 파이프라인 전체(로드→추론→판정→AI 대결→렌더)를 검증할 수
있어야 하므로 --image 모드가 반드시 있어야 한다.

실행:
    ../../../venv/bin/python rps_game.py --image photo.jpg     # 정지 이미지 모드
    ../../../venv/bin/python rps_game.py                       # 웹캠 모드 (직접 실행 금지 — 사용자가 검증)

조작(웹캠 모드): [Q] 종료   [R] 점수 초기화
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from hand_lite.rps import AIMoveProvider, Game, MoveStabilizer

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "hand_landmarker.task"
WINDOW = "MediaPipe RPS vs AI"

# app.py와 동일한 이유로 프레임 인덱스 기반 타임스탬프를 쓴다(VIDEO 모드
# 단조 증가 제약 — hand_lite/landmarker.py 참고).
FRAME_INTERVAL_MS = 33


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="경량 MediaPipe 가위바위보 vs AI")
    p.add_argument("--image", default=None,
                   help="정지 이미지 경로. 지정하면 웹캠 대신 이 이미지 1장으로 라운드 1회를 판정한다")
    p.add_argument("--out", default="output", help="이미지 모드 결과 저장 디렉터리. 기본값: output/")
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="HandLandmarker .task 모델 경로")
    p.add_argument("--seed", type=int, default=None,
                   help="AI 패 난수 시드. 지정하면 AI가 항상 같은 순서로 낸다(테스트/데모 재현용)")
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
    """카메라 없이 파이프라인(로드→추론→판정→AI 대결→렌더) 전체를 정지 이미지
    1장으로 검증한다.

    정지 이미지는 프레임이 1장뿐이라 지터가 없다 — 그래서 안정화(MoveStabilizer)
    없이 classify() 결과를 바로 라운드에 쓴다. 안정화는 여러 프레임에 걸친
    흔들림을 걸러내는 장치라 단일 프레임에는 적용 대상 자체가 없다.
    """
    from hand_lite.gesture import classify
    from hand_lite.hud import draw_hand, draw_label, draw_scoreboard, draw_round_status
    from hand_lite.landmarker import HandLite

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"이미지 파일이 없습니다: {image_path}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        sys.exit(f"이미지를 읽지 못했습니다(지원 형식인지 확인하세요): {image_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_rps_result.png"

    with HandLite(model_path=args.model, num_hands=1) as landmarker:
        hands = landmarker.detect(frame, timestamp_ms=0)

    game = Game(ai_provider=AIMoveProvider(seed=args.seed))
    print(f"[검출] 손 {len(hands)}개")

    if not hands:
        status_text = "손을 찾지 못했습니다"
        print(f"[판정] {status_text}")
    else:
        hand = hands[0]  # 1인용 대전 — 첫 번째로 검출된 손만 플레이어로 취급한다
        gesture = classify(hand)
        print(f"  손: {hand.handedness}, {gesture.value} (score={hand.score:.2f})")
        round_state = game.play_round(gesture)
        draw_hand(frame, hand)
        draw_label(frame, hand, gesture)

        w, l, d = game.score.as_tuple()
        if round_state.result.value == "판정불가":
            status_text = "판정불가 — 가위/바위/보 중 하나를 명확히 내주세요"
        else:
            status_text = f"플레이어 {round_state.player.value} vs AI {round_state.ai.value} → {round_state.result.value}"
        print(f"[판정] {status_text}")
        print(f"[전적] {w}승 {l}패 {d}무")

    draw_scoreboard(frame, *game.score.as_tuple())
    draw_round_status(frame, status_text)

    cv2.imwrite(str(out_path), frame)
    print(f"[완료] 결과 저장: {out_path}")


def run_webcam_mode(args: argparse.Namespace) -> None:
    """실시간 대결 루프.

    라운드 트리거 방식: MoveStabilizer가 새로운 손모양을 '확정'하는 순간(직전
    확정값과 다를 때) 1회만 라운드를 진행한다. 확정 상태가 풀렸다가(손을 바꾸는
    중 등) 다시 같은 손모양으로 확정돼야 재대결이 가능하다 — 손을 그대로 들고
    있는데 매 프레임 판정이 반복되는 것을 막기 위함이다.
    """
    from hand_lite.gesture import classify
    from hand_lite.hud import draw_hand, draw_label, draw_scoreboard, draw_round_status
    from hand_lite.landmarker import HandLite

    cap = open_capture()
    landmarker = HandLite(model_path=args.model, num_hands=1)
    stabilizer = MoveStabilizer()
    game = Game(ai_provider=AIMoveProvider(seed=args.seed))
    last_played = None  # 직전에 라운드로 소비된 확정 손모양 — 중복 대결 방지

    print("조작: [Q] 종료   [R] 점수 초기화\n")

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("카메라 프레임을 읽지 못했습니다. 종료합니다.")
                break

            frame = cv2.flip(frame, 1)  # 거울 모드 — gesture.py는 이제 이 미러링에 영향받지 않는다

            timestamp_ms = frame_index * FRAME_INTERVAL_MS
            hands = landmarker.detect(frame, timestamp_ms)
            frame_index += 1

            current_gesture = None
            if hands:
                hand = hands[0]
                current_gesture = classify(hand)
                draw_hand(frame, hand)
                draw_label(frame, hand, current_gesture)

            stabilizer.update(current_gesture) if current_gesture is not None else stabilizer.clear()
            stable = stabilizer.stable_gesture()

            status_text = "손을 보여주세요"
            if stable is not None and stable != last_played:
                round_state = game.play_round(stable)
                last_played = stable
                if round_state.result.value == "판정불가":
                    status_text = "판정불가 — 다시 내주세요"
                else:
                    status_text = f"{round_state.player.value} vs {round_state.ai.value} → {round_state.result.value}"
            elif stable is None:
                last_played = None  # 확정이 풀렸으니 다음 확정은 새 라운드로 인정
                status_text = "인식 중..."

            draw_scoreboard(frame, *game.score.as_tuple())
            draw_round_status(frame, status_text)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                game.reset_score()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        w, l, d = game.score.as_tuple()
        print(f"\n최종 전적: {w}승 {l}패 {d}무")


def main() -> None:
    args = parse_args()
    if args.image:
        run_image_mode(args)
    else:
        run_webcam_mode(args)


if __name__ == "__main__":
    main()
