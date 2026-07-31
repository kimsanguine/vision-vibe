"""cv2 화면 구성 — PIL로 한글/이모지를 그린 뒤 BGR 프레임으로 되돌린다.

cv2.putText는 한글을 렌더링하지 못하므로(폰트에 CJK 글리프 없음)
PIL ImageDraw로 캔버스를 그리고 마지막에 numpy BGR로 변환한다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .judge import EMOJI_FALLBACK, KOREAN, OUTCOME_KOREAN, Move, Outcome

CANVAS_W, CANVAS_H = 1180, 680
HEADER_H = 70
CAM_BOX = (20, 90, 640, 480)   # x, y, w, h
AI_BOX = (680, 90, 480, 480)
FOOTER_Y = 590

BG = (18, 23, 43)
PANEL = (28, 35, 64)
TEXT = (242, 245, 255)
MUTED = (138, 147, 178)
ACCENT = (255, 122, 61)
BLUE = (77, 163, 255)
OUTCOME_COLOR = {
    Outcome.WIN: (61, 220, 132),
    Outcome.LOSE: (255, 92, 122),
    Outcome.DRAW: (255, 201, 61),
}

_KO_FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
_KO_FONT_ALT = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
_EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"
# Apple Color Emoji는 비트맵 폰트라 특정 크기(…, 96, 160)만 로드된다.
_EMOJI_NATIVE_SIZE = 160


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in (_KO_FONT, _KO_FONT_ALT):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


class Fonts:
    def __init__(self) -> None:
        self.title = _load_font(30)
        self.score = _load_font(26)
        self.label = _load_font(24)
        self.status = _load_font(40)
        self.count = _load_font(150)
        self.footer = _load_font(18)
        try:
            self.emoji = ImageFont.truetype(_EMOJI_FONT, _EMOJI_NATIVE_SIZE)
        except OSError:
            self.emoji = None


def _center_text(draw, box, text, font, fill) -> None:
    x, y, w, h = box
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text((x + (w - (right - left)) / 2 - left, y + (h - (bottom - top)) / 2 - top),
              text, font=font, fill=fill)


def _fit_into(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """비율을 유지한 채 w×h 안에 letterbox로 맞춘다."""
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), PANEL[::-1], dtype=np.uint8)  # BGR
    ox, oy = (w - nw) // 2, (h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def draw_detection_box(frame: np.ndarray, box, move: Move, confidence: float) -> None:
    """카메라 프레임 위에 검출 박스를 그린다(라틴 문자만 사용 — cv2.putText 제약)."""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (61, 220, 61), 3)
    caption = f"{move.value} {confidence:.2f}"
    cv2.putText(frame, caption, (x1, max(24, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (61, 220, 61), 2, cv2.LINE_AA)


class HUD:
    def __init__(self) -> None:
        self.fonts = Fonts()
        self._ai_image_cache: dict[Path, np.ndarray] = {}

    def _ai_image(self, path: Path) -> np.ndarray:
        if path not in self._ai_image_cache:
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(f"AI 패 이미지를 읽지 못했습니다: {path}")
            self._ai_image_cache[path] = img
        return self._ai_image_cache[path]

    def render(
        self,
        camera_frame: np.ndarray,
        *,
        score: tuple[int, int, int],
        status_text: str,
        status_color: tuple[int, int, int] = TEXT,
        countdown: int | None = None,
        live_move: Move | None = None,
        user_move: Move | None = None,
        ai_move: Move | None = None,
        ai_image_path: Path | None = None,
        footer_text: str = "",
    ) -> np.ndarray:
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        draw = ImageDraw.Draw(canvas)

        # --- 헤더 ---
        draw.rectangle([0, 0, CANVAS_W, HEADER_H], fill=PANEL)
        draw.text((24, 20), "YOLO 가위바위보 · vs AI", font=self.fonts.title, fill=TEXT)
        wins, losses, draws = score
        score_text = f"{wins}승  {losses}패  {draws}무"
        bbox = draw.textbbox((0, 0), score_text, font=self.fonts.score)
        draw.text((CANVAS_W - 24 - (bbox[2] - bbox[0]), 22), score_text,
                  font=self.fonts.score, fill=ACCENT)

        # --- 카메라 패널 ---
        cx, cy, cw, ch = CAM_BOX
        fitted = _fit_into(camera_frame, cw, ch)
        canvas.paste(Image.fromarray(cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB)), (cx, cy))
        draw.rectangle([cx - 2, cy - 2, cx + cw + 2, cy + ch + 2], outline=BLUE, width=2)
        draw.text((cx, cy - 32), "당신", font=self.fonts.label, fill=BLUE)

        live_label = KOREAN[live_move] if live_move else "손을 보여주세요"
        draw.text((cx + 90, cy - 30), f"인식: {live_label}",
                  font=self.fonts.label, fill=TEXT if live_move else MUTED)

        if countdown is not None:
            _center_text(draw, (cx, cy, cw, ch), str(countdown), self.fonts.count, ACCENT)

        # --- AI 패널 ---
        ax, ay, aw, ah = AI_BOX
        draw.rectangle([ax, ay, ax + aw, ay + ah], fill=PANEL)
        draw.rectangle([ax - 2, ay - 2, ax + aw + 2, ay + ah + 2], outline=ACCENT, width=2)
        draw.text((ax, ay - 32), "AI", font=self.fonts.label, fill=ACCENT)

        if ai_move is None:
            _center_text(draw, (ax, ay, aw, ah), "?", self.fonts.count, MUTED)
        elif ai_image_path is not None:
            img = self._ai_image(ai_image_path)
            fitted_ai = _fit_into(img, aw - 20, ah - 70)  # 하단 50px는 라벨 띠 자리
            canvas.paste(Image.fromarray(cv2.cvtColor(fitted_ai, cv2.COLOR_BGR2RGB)),
                         (ax + 10, ay + 10))
        else:
            self._draw_emoji(canvas, draw, (ax, ay, aw, ah), ai_move)

        if ai_move is not None:
            # 이미지 위에 그대로 얹으면 글자가 묻히므로 어두운 띠를 깔아준다.
            draw.rectangle([ax, ay + ah - 50, ax + aw, ay + ah], fill=BG)
            _center_text(draw, (ax, ay + ah - 50, aw, 50), KOREAN[ai_move],
                         self.fonts.label, TEXT)

        # --- 푸터 ---
        draw.rectangle([0, FOOTER_Y, CANVAS_W, CANVAS_H], fill=PANEL)
        _center_text(draw, (0, FOOTER_Y, CANVAS_W, 52), status_text,
                     self.fonts.status, status_color)
        if user_move and ai_move:
            detail = f"당신 {KOREAN[user_move]}  vs  AI {KOREAN[ai_move]}"
            _center_text(draw, (0, FOOTER_Y + 50, CANVAS_W, 22), detail,
                         self.fonts.footer, MUTED)
        elif footer_text:
            _center_text(draw, (0, FOOTER_Y + 50, CANVAS_W, 22), footer_text,
                         self.fonts.footer, MUTED)

        return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

    def _draw_emoji(self, canvas: Image.Image, draw, box, move: Move) -> None:
        """이미지 폴백 — 이모지를 네이티브 크기로 그린 뒤 패널에 맞춰 확대."""
        x, y, w, h = box
        if self.fonts.emoji is None:
            _center_text(draw, box, KOREAN[move], self.fonts.count, TEXT)
            return
        s = _EMOJI_NATIVE_SIZE
        tile = Image.new("RGBA", (s * 2, s * 2), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((0, 0), EMOJI_FALLBACK[move],
                                  font=self.fonts.emoji, embedded_color=True)
        tile = tile.crop(tile.getbbox() or (0, 0, s, s))
        target = int(min(w, h) * 0.6)
        tile = tile.resize((target, target), Image.LANCZOS)
        canvas.paste(tile, (x + (w - target) // 2, y + (h - target) // 2 - 20), tile)
