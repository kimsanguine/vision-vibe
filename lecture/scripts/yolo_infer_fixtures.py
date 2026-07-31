"""yolo1_rps venv에서 실행되는 헬퍼 — 6장 실사진 픽스처에 YOLO 실측을 돌린다.

왜 별도 스크립트인가
────────────────────
이 워크샵 저장소는 프로젝트마다 독립된 venv를 쓴다(각자 다른 무거운 의존성 —
mediapipe vs torch/ultralytics — 를 설치하므로 하나로 합칠 수 없다). 노트북은
루트 venv(opencv+mediapipe) 커널로 돌아가므로, YOLO(ultralytics/torch)만은
`yolo1_rps/venv/bin/python`으로 이 스크립트를 서브프로세스 실행해 결과를
JSON으로 돌려받는다. yolo1_rps 디렉토리 자체는 전혀 수정하지 않는다 — 이미
있는 `rps/detector.py`·`rps/judge.py`를 import해서 쓰기만 한다.

사용법(노트북에서):
    subprocess.run([yolo_venv_python, this_script, fixtures_dir, model_path],
                    capture_output=True, text=True)
표준출력에 JSON 한 줄을 찍는다: {"rock_0.png": {"move": "rock", "confidence": 0.91}, ...}
검출 실패(손 없음/클래스 매핑 실패)면 값이 null.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

# yolo1_rps 패키지(rps/)를 import 하기 위해 그 프로젝트 루트를 sys.path에 추가한다.
# yolo1_rps 자체 파일은 하나도 건드리지 않는다 — 읽어서 import만 한다.
YOLO_PROJECT_ROOT = Path(__file__).resolve().parents[2] / "src" / "yolo" / "rps"
sys.path.insert(0, str(YOLO_PROJECT_ROOT))

from rps.detector import RPSDetector  # noqa: E402


def main() -> None:
    fixtures_dir = Path(sys.argv[1])
    model_path = Path(sys.argv[2])

    detector = RPSDetector(model_path=model_path, device="cpu")
    # device="cpu" 고정 이유: mps 자동선택은 기기마다 가용성이 달라 재현성이
    # 떨어진다. 이 스크립트의 목적은 "검출되는가/신뢰도가 얼마인가"의 실측이지
    # 추론 속도 벤치마크가 아니므로 가장 이식성 높은 cpu로 고정한다.

    results: dict[str, dict | None] = {}
    for image_path in sorted(fixtures_dir.glob("*.png")):
        frame = cv2.imread(str(image_path))
        detection = detector.detect(frame)
        if detection is None:
            results[image_path.name] = None
        else:
            results[image_path.name] = {
                "move": detection.move.value,
                "confidence": round(detection.confidence, 4),
            }

    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
