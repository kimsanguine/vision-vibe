"""웹캠 데모 — 전통 영상처리 가위바위보.

[검증 상태] 이 파일은 **이 환경에서 실행 검증되지 않았다.** 웹캠 캡처가
불가능한 환경이라 cv2.VideoCapture 이후의 경로를 한 번도 실제로 돌려보지
못했다. 픽스처 6장 경로(evaluate.py)와 단위 테스트만 검증된 상태다.
로직 자체는 검증된 rps_classic 패키지를 그대로 호출하므로 판정 결과는
evaluate.py 와 동일하지만, "카메라가 열리는가 / 키 입력이 먹는가 / 프레임률이
쓸 만한가"는 사용자가 직접 확인해야 한다.

ROI(관심영역) 상자를 두는 이유
──────────────────────────────
전체 화면에 피부색 임계값을 걸면 얼굴·팔·나무 책상·베이지 벽이 전부 손과 함께
마스킹되고, "가장 큰 컨투어 = 손" 가정이 즉시 무너진다. 실습에서 이 실패를
보여주는 것도 의미 있지만, 게임을 굴리려면 손을 넣을 영역을 고정하는 편이
현실적이다. 이 상자 자체가 전통 기법의 한계를 드러내는 장치다 —
YOLO/MediaPipe 는 이런 상자를 요구하지 않는다.

실행: ../../../venv/bin/python app.py
조작: [스페이스] 라운드 진행  [m] 마스크 보기 토글  [r] 점수 초기화  [q] 종료
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rps_classic.judge import AIMoveProvider, judge  # noqa: E402
from rps_classic.pipeline import detect  # noqa: E402
from rps_classic.types import Gesture, Result  # noqa: E402

CAMERA_INDEX = 0
ROI = (320, 80, 280, 280)          # (x, y, w, h) — 손을 넣을 영역
WINDOW = "RPS - OpenCV Classic"

# 한글은 cv2.putText 로 그릴 수 없다(폰트 미지원 — 물음표로 깨진다).
# 별도 폰트 렌더링을 붙이는 대신 영문/기호로 표기한다(Rule 2 — 최소 코드).
GESTURE_LABEL = {
    Gesture.ROCK: "ROCK",
    Gesture.PAPER: "PAPER",
    Gesture.SCISSORS: "SCISSORS",
    Gesture.UNKNOWN: "???",
}
RESULT_LABEL = {
    Result.WIN: "YOU WIN",
    Result.LOSE: "YOU LOSE",
    Result.DRAW: "DRAW",
    Result.UNDECIDABLE: "NO CALL (unclear hand)",
}


def draw_hud(frame, detection, last_text: str, score: tuple[int, int, int]) -> None:
    x, y, w, h = ROI
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if detection.contour is not None:
        # ROI 좌표계의 컨투어를 전체 프레임 좌표로 옮겨 그린다.
        shifted = detection.contour + (x, y)
        cv2.drawContours(frame, [shifted], -1, (0, 255, 0), 2)
        cv2.drawContours(frame, [cv2.convexHull(shifted)], -1, (255, 0, 0), 2)

    shape = detection.shape
    lines = [
        f"gesture: {GESTURE_LABEL[detection.gesture]}",
        "hand: not found" if shape is None
        else f"gaps={shape.finger_gaps} fingers={shape.fingers} solidity={shape.solidity:.2f}",
        f"score W/L/D: {score[0]}/{score[1]}/{score[2]}",
        last_text,
        "[space] play  [m] mask  [r] reset  [q] quit",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (12, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (12, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> int:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[중단] 카메라 {CAMERA_INDEX} 를 열지 못했다. "
              f"다른 앱이 점유 중이거나 권한이 없을 수 있다.")
        return 1

    ai = AIMoveProvider()
    wins = losses = draws = 0
    last_text = "press [space] to play"
    show_mask = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[중단] 프레임을 읽지 못했다.")
                return 1

            # 거울 모드 — 사용자가 오른손을 들면 화면에서도 오른쪽에 보여야
            # 손을 ROI 에 넣기 쉽다.
            frame = cv2.flip(frame, 1)

            x, y, w, h = ROI
            roi = frame[y:y + h, x:x + w]
            detection = detect(roi)

            draw_hud(frame, detection, last_text, (wins, losses, draws))
            if show_mask:
                # 마스크를 ROI 자리에 겹쳐 보여준다 — 임계값이 무엇을 잡고
                # 무엇을 놓치는지 실시간으로 보는 것이 이 실습의 핵심이다.
                frame[y:y + h, x:x + w] = cv2.cvtColor(detection.mask, cv2.COLOR_GRAY2BGR)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("m"):
                show_mask = not show_mask
            elif key == ord("r"):
                wins = losses = draws = 0
                last_text = "score reset"
            elif key == ord(" "):
                if detection.gesture is Gesture.UNKNOWN:
                    # 억지로 판정하지 않는다. 왜 기권했는지 원인을 함께 보여준다.
                    reason = "no hand in ROI" if detection.shape is None \
                        else f"ambiguous ({detection.shape.fingers} fingers)"
                    last_text = f"NO CALL - {reason}"
                else:
                    ai_move = ai.pick()
                    result = judge(detection.gesture, ai_move)
                    if result is Result.WIN:
                        wins += 1
                    elif result is Result.LOSE:
                        losses += 1
                    elif result is Result.DRAW:
                        draws += 1
                    last_text = (f"you {GESTURE_LABEL[detection.gesture]} vs "
                                 f"ai {GESTURE_LABEL[ai_move]} -> {RESULT_LABEL[result]}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
