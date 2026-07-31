"""마법 지팡이 (Magic Wand) — OpenCV 고전 CV만으로 만드는 색상 추적 궤적 앱.

딥러닝 없이 HSV 색상 마스킹 + 컨투어 무게중심만으로 물체를 추적하고,
페이드아웃 / 무지개 궤적 / 글로우 / 헤일로로 '마법' 연출을 얹는다.

실행:  python magic_wand.py
조작:  [s] 캔버스 지우기   [c] 추적 색상 변경   [q] 종료
"""

import random

import cv2
import numpy as np

# ── 설정 상수 ────────────────────────────────────────────────
CAMERA_INDEX = 0
MIRROR = True          # 거울 모드 (지팡이 조작 직관성)

MIN_AREA = 300         # 이보다 작은 컨투어는 노이즈로 간주 (픽셀 면적)
FADE = 0.93            # 매 프레임 캔버스 감쇠율 (낮을수록 궤적이 빨리 사라짐)
TRAIL_THICKNESS = 5
MAX_JUMP = 200         # 프레임 간 이동이 이보다 크면 선을 잇지 않음 (오검출 방어)

GLOW_SIGMA = 9         # 글로우 블러 강도
GLOW_GAIN = 0.9        # 글로우 합성 비율

HUE_STEP = 3           # 프레임당 무지개 색상 진행량 (H: 0~179)
HALO_RADIUS = 22
SPARKLE_PER_FRAME = 2
SPARKLE_LIFE = 12

# OpenCV HSV 범위: H 0~179, S 0~255, V 0~255.
# 빨강은 색상환 0도 부근이라 구간이 끊겨 두 개의 범위가 필요하다.
COLOR_PRESETS = [
    {
        "name": "RED",
        "ranges": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
        "hud": (60, 60, 255),
    },
    {
        "name": "BLUE",
        "ranges": [((100, 120, 70), (130, 255, 255))],
        "hud": (255, 140, 60),
    },
    {
        "name": "GREEN",
        "ranges": [((40, 90, 70), (80, 255, 255))],
        "hud": (80, 255, 120),
    },
    {
        "name": "YELLOW",
        "ranges": [((22, 120, 100), (35, 255, 255))],
        "hud": (60, 230, 255),
    },
]

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ── 1. 색상 추적 (HSV → inRange → findContours → 무게중심) ────
def build_mask(hsv, preset):
    """HSV 프레임에서 프리셋 색상만 남긴 이진 마스크를 만든다."""
    mask = None
    for lo, hi in preset["ranges"]:
        part = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    # 열림 연산으로 점 노이즈 제거 → 닫힘 연산으로 물체 내부 구멍 메우기
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    return mask


def find_centroid(mask, min_area=MIN_AREA):
    """마스크에서 가장 큰 컨투어의 무게중심 (x, y)를 반환. 없으면 None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None
    return (int(round(m["m10"] / m["m00"])), int(round(m["m01"] / m["m00"])))


def track(frame_bgr, preset, min_area=MIN_AREA):
    """BGR 프레임 하나에서 지팡이 끝 좌표를 추적한다. (테스트 진입점)"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return find_centroid(build_mask(hsv, preset), min_area)


# ── 2. 마법 연출 ────────────────────────────────────────────
def fade_canvas(canvas, factor=FADE):
    """캔버스를 곱셈으로 어둡게 해 오래된 궤적이 서서히 사라지게 한다."""
    return (canvas.astype(np.float32) * factor).astype(np.uint8)


def rainbow_color(tick):
    """프레임 번호에 따라 순환하는 무지개 BGR 색상."""
    hue = (tick * HUE_STEP) % 180
    pixel = np.uint8([[[hue, 255, 255]]])
    b, g, r = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0][0]
    return (int(b), int(g), int(r))


def apply_glow(canvas, sigma=GLOW_SIGMA, gain=GLOW_GAIN):
    """궤적 주변으로 빛이 번지는 글로우 레이어를 만든다 (캔버스 자체는 불변)."""
    blur = cv2.GaussianBlur(canvas, (0, 0), sigma)
    return cv2.addWeighted(canvas, 1.0, blur, gain, 0)


def draw_halo(layer, center, tick, color):
    """지팡이 끝을 감싸는 맥동하는 헤일로."""
    pulse = int(HALO_RADIUS + 6 * np.sin(tick * 0.25))
    cv2.circle(layer, center, pulse, color, 2, cv2.LINE_AA)
    cv2.circle(layer, center, max(pulse - 8, 2), color, 1, cv2.LINE_AA)
    cv2.circle(layer, center, 3, (255, 255, 255), -1, cv2.LINE_AA)


def spawn_sparkles(sparkles, center):
    """지팡이 끝에서 반짝이 파티클을 뿌린다."""
    for _ in range(SPARKLE_PER_FRAME):
        sparkles.append(
            {
                "pos": [float(center[0]), float(center[1])],
                "vel": [random.uniform(-2.5, 2.5), random.uniform(-2.5, 2.5)],
                "life": SPARKLE_LIFE,
            }
        )


def update_sparkles(sparkles, layer):
    """파티클을 이동시키며 그리고, 수명이 다한 것은 제거한다."""
    alive = []
    for s in sparkles:
        s["pos"][0] += s["vel"][0]
        s["pos"][1] += s["vel"][1]
        s["life"] -= 1
        if s["life"] <= 0:
            continue
        shade = int(255 * s["life"] / SPARKLE_LIFE)
        cv2.circle(
            layer,
            (int(s["pos"][0]), int(s["pos"][1])),
            2,
            (shade, shade, 255),
            -1,
            cv2.LINE_AA,
        )
        alive.append(s)
    sparkles[:] = alive


def draw_hud(frame, preset, detected):
    """마법 지팡이 컨셉 HUD."""
    h = frame.shape[0]
    cv2.putText(frame, "* MAGIC WAND *", (16, 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"wand color: {preset['name']}", (16, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, preset["hud"], 2, cv2.LINE_AA)

    status = "WAND DETECTED" if detected else "SHOW YOUR WAND"
    status_color = (120, 255, 120) if detected else (120, 120, 255)
    cv2.putText(frame, status, (16, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

    cv2.putText(frame, "[s] clear   [c] color   [q] quit", (16, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)


def compose(frame, canvas, center, tick, preset, sparkles):
    """프레임 + 궤적(글로우) + 헤일로/파티클 + HUD 를 합성한다."""
    effect = apply_glow(canvas)
    fx_layer = np.zeros_like(frame)

    if center is not None:
        draw_halo(fx_layer, center, tick, rainbow_color(tick))
    update_sparkles(sparkles, fx_layer)

    out = cv2.add(frame, effect)
    out = cv2.add(out, fx_layer)
    draw_hud(out, preset, center is not None)
    return out


# ── 3. 메인 루프 ────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"카메라를 열 수 없습니다 (index={CAMERA_INDEX}).")

    canvas = None
    prev = None
    sparkles = []
    preset_idx = 0
    tick = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if MIRROR:
            frame = cv2.flip(frame, 1)
        if canvas is None:
            canvas = np.zeros_like(frame)

        preset = COLOR_PRESETS[preset_idx]
        center = track(frame, preset)

        canvas = fade_canvas(canvas)
        if center is not None:
            if prev is not None and np.hypot(center[0] - prev[0], center[1] - prev[1]) < MAX_JUMP:
                cv2.line(canvas, prev, center, rainbow_color(tick),
                         TRAIL_THICKNESS, cv2.LINE_AA)
            spawn_sparkles(sparkles, center)
        prev = center

        cv2.imshow("Magic Wand", compose(frame, canvas, center, tick, preset, sparkles))
        tick += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            canvas[:] = 0
            sparkles.clear()
        if key == ord("c"):
            preset_idx = (preset_idx + 1) % len(COLOR_PRESETS)
            prev = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
