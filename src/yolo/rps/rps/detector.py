"""YOLO 기반 가위바위보 손모양 인식."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import torch
from ultralytics import YOLO

from .judge import Move, move_from_class_name


@dataclass
class Detection:
    move: Move
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2


def pick_device(requested: str | None = None) -> str:
    """추론 디바이스 선택. Apple Silicon에서는 mps가 CPU 대비 약 4배 빠르다."""
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class RPSDetector:
    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        imgsz: int = 640,
        conf: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.device = pick_device(device)
        self.imgsz = imgsz
        self.conf = conf
        self.class_names: dict[int, str] = self.model.names

    def detect(self, frame) -> Detection | None:
        """프레임에서 가장 신뢰도 높은 손모양 1개를 반환. 없으면 None."""
        result = self.model.predict(
            frame,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False,
        )[0]
        if not len(result.boxes):
            return None

        best = max(result.boxes, key=lambda b: float(b.conf))
        move = move_from_class_name(self.class_names[int(best.cls)])
        if move is None:
            return None

        x1, y1, x2, y2 = (int(v) for v in best.xyxy[0].tolist())
        return Detection(move=move, confidence=float(best.conf), box=(x1, y1, x2, y2))


class MoveStabilizer:
    """프레임 단위 흔들림 제거 — 최근 N프레임의 다수결로 손모양을 확정한다.

    단일 프레임 결과를 그대로 쓰면 손을 바꾸는 순간의 노이즈가 판정에 섞인다.
    """

    def __init__(self, window: int = 7, min_votes: int = 4) -> None:
        self.buffer: deque[Move | None] = deque(maxlen=window)
        self.min_votes = min_votes

    def update(self, move: Move | None) -> None:
        self.buffer.append(move)

    def stable_move(self) -> Move | None:
        votes = Counter(m for m in self.buffer if m is not None)
        if not votes:
            return None
        move, count = votes.most_common(1)[0]
        return move if count >= self.min_votes else None

    def clear(self) -> None:
        self.buffer.clear()
