"""합성 데이터 검증 — 웹캠 없이 추적 로직과 연출을 수치로 확인한다.

numpy로 '알려진 경로'를 따라 움직이는 색 원을 만들고,
magic_wand의 추적 결과가 그 정답 좌표와 5픽셀 이내로 일치하는지 검사한다.

실행: python test_synthetic.py   (결과 이미지는 output/ 에 저장)
"""

import os

import cv2
import numpy as np

import magic_wand as mw

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
W, H = 640, 480
RADIUS = 25
TOLERANCE = 5.0  # 픽셀

# 프리셋 이름 → 합성용 대표 BGR (해당 HSV 범위 한가운데 색)
SAMPLE_BGR = {"RED": (0, 0, 255), "BLUE": (255, 0, 0),
              "GREEN": (0, 200, 0), "YELLOW": (0, 220, 235)}


def preset_by_name(name):
    return next(p for p in mw.COLOR_PRESETS if p["name"] == name)


def make_frame(center, bgr, radius=RADIUS, noise=True, extras=()):
    """회색 배경 + 지정 색 원(+옵션 방해물)로 이루어진 합성 프레임."""
    frame = np.full((H, W, 3), 40, np.uint8)
    if noise:
        rng = np.random.default_rng(0)
        frame = cv2.add(frame, rng.integers(0, 12, (H, W, 3), dtype=np.uint8))
    for ex_center, ex_radius in extras:
        cv2.circle(frame, ex_center, ex_radius, bgr, -1)
    cv2.circle(frame, center, radius, bgr, -1)
    return frame


def sine_path(n=8):
    """정답 경로: 좌→우로 이동하며 사인 곡선을 그린다."""
    return [(80 + i * 60, int(240 + 90 * np.sin(i * 0.7))) for i in range(n)]


# ── 검증 1: 프레임별 무게중심 정확도 ──────────────────────────
def test_centroid_accuracy():
    print("\n[1] 프레임별 무게중심 정확도 (목표: 오차 < 5px)")
    all_errors = []
    for name in ("RED", "BLUE", "GREEN", "YELLOW"):
        preset, bgr = preset_by_name(name), SAMPLE_BGR[name]
        errors = []
        for truth in sine_path():
            got = mw.track(make_frame(truth, bgr), preset)
            assert got is not None, f"{name}: 프레임 {truth} 에서 물체 미검출"
            err = float(np.hypot(got[0] - truth[0], got[1] - truth[1]))
            errors.append(err)
        all_errors += errors
        print(f"    {name:<7} 프레임 {len(errors)}개 | 평균 {np.mean(errors):.3f}px "
              f"| 최대 {max(errors):.3f}px")
        assert max(errors) < TOLERANCE, f"{name} 오차 {max(errors):.2f}px > {TOLERANCE}px"
    print(f"    → 전체 {len(all_errors)}개 측정, 최대 오차 {max(all_errors):.3f}px  PASS")


# ── 검증 2: 오검출 방어 (더 작은 방해물 / 물체 없음) ──────────
def test_robustness():
    print("\n[2] 오검출 방어")
    preset, bgr = preset_by_name("RED"), SAMPLE_BGR["RED"]

    truth = (400, 300)
    frame = make_frame(truth, bgr, extras=[((120, 120), 14)])  # 작은 동색 방해물
    got = mw.track(frame, preset)
    err = float(np.hypot(got[0] - truth[0], got[1] - truth[1]))
    print(f"    작은 방해물 존재 시 큰 물체 선택: 오차 {err:.3f}px")
    assert err < TOLERANCE

    empty = make_frame((0, 0), bgr, radius=0)
    print(f"    물체 없는 프레임 결과: {mw.track(empty, preset)}")
    assert mw.track(empty, preset) is None

    tiny = make_frame((320, 240), bgr, radius=5)  # 면적 ≈ 79 < MIN_AREA(300)
    print(f"    MIN_AREA 미만(r=5) 결과: {mw.track(tiny, preset)}")
    assert mw.track(tiny, preset) is None
    print("    → PASS")


# ── 검증 3: 이은 궤적이 의도한 경로와 일치하는가 ──────────────
def test_trail_matches_path():
    print("\n[3] 궤적-경로 일치 (캔버스에 그려진 선이 정답 경로 위에 있는가)")
    preset, bgr = preset_by_name("RED"), SAMPLE_BGR["RED"]
    path = sine_path()
    canvas = np.zeros((H, W, 3), np.uint8)
    prev = None
    tracked = []

    for tick, truth in enumerate(path):
        center = mw.track(make_frame(truth, bgr), preset)
        tracked.append(center)
        if prev is not None:
            cv2.line(canvas, prev, center, mw.rainbow_color(tick), mw.TRAIL_THICKNESS)
        prev = center

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    r = int(TOLERANCE)
    distances = []
    for x, y in path:
        window = gray[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
        assert window.max() > 0, f"경로점 {(x, y)} 주변 {r}px 안에 궤적 없음"
        ys, xs = np.nonzero(window)
        d = np.hypot(xs + max(x - r, 0) - x, ys + max(y - r, 0) - y).min()
        distances.append(float(d))
    print(f"    경로점 {len(path)}개 → 궤적까지 최단거리 평균 {np.mean(distances):.3f}px, "
          f"최대 {max(distances):.3f}px")
    assert max(distances) < TOLERANCE
    cv2.imwrite(os.path.join(OUT_DIR, "verify_trail_path.png"), canvas)
    print("    → PASS (output/verify_trail_path.png 저장)")
    return canvas


# ── 검증 4: 페이드아웃이 실제로 픽셀값을 낮추는가 ─────────────
def test_fade():
    print(f"\n[4] 페이드아웃 (FADE={mw.FADE})")
    canvas = np.zeros((H, W, 3), np.uint8)
    cv2.line(canvas, (100, 240), (540, 240), (255, 255, 255), mw.TRAIL_THICKNESS)
    base = float(canvas.max())

    snapshots, values = {0: canvas.copy()}, [base]
    for step in range(1, 61):
        canvas = mw.fade_canvas(canvas)
        values.append(float(canvas.max()))
        if step in (5, 15, 30, 60):
            snapshots[step] = canvas.copy()

    for step in sorted(snapshots):
        expected = 255 * (mw.FADE ** step)
        print(f"    {step:>2}프레임 후 최대 픽셀값 {values[step]:6.1f} "
              f"(이론값 {expected:6.1f})")
        cv2.imwrite(os.path.join(OUT_DIR, f"verify_fade_{step:02d}.png"), snapshots[step])

    # 0에 도달한 뒤에는 계속 0이므로, 양수인 구간에서만 '엄격히 감소'를 요구한다.
    assert all(values[i] > values[i + 1]
               for i in range(len(values) - 1) if values[i] > 0), "0에 닿기 전 정체 구간 존재"
    assert values[60] == 0, f"60프레임 후에도 잔상 {values[60]}"
    print("    → PASS (양수 구간 엄격 감소 + 60프레임 내 완전 소멸, "
          "output/verify_fade_*.png 저장)")


# ── 검증 5: 글로우가 주변으로 빛을 번지게 하는가 ──────────────
def test_glow():
    print(f"\n[5] 글로우 (sigma={mw.GLOW_SIGMA}, gain={mw.GLOW_GAIN})")
    canvas = np.zeros((H, W, 3), np.uint8)
    cv2.line(canvas, (100, 240), (540, 240), (255, 255, 255), mw.TRAIL_THICKNESS)
    glow = mw.apply_glow(canvas)

    for dy in (0, 5, 10, 20):
        before, after = int(canvas[240 + dy, 320].max()), int(glow[240 + dy, 320].max())
        print(f"    선 중심에서 {dy:>2}px 떨어진 픽셀: {before:>3} → {after:>3}")
        if dy > mw.TRAIL_THICKNESS:
            assert before == 0 and after > 0, f"{dy}px 지점에 빛 번짐 없음"
    cv2.imwrite(os.path.join(OUT_DIR, "verify_glow.png"), glow)
    print("    → PASS (궤적 밖 픽셀이 0에서 양수로, output/verify_glow.png 저장)")


# ── 검증 6: 전체 파이프라인 합성 (눈으로 확인할 데모 프레임) ──
def test_full_pipeline_render():
    print("\n[6] 전체 합성 렌더 (추적 + 페이드 + 무지개 + 글로우 + 헤일로 + HUD)")
    preset, bgr = preset_by_name("RED"), SAMPLE_BGR["RED"]
    path = [(70 + i * 22, int(240 + 110 * np.sin(i * 0.28))) for i in range(26)]

    canvas = np.zeros((H, W, 3), np.uint8)
    prev, sparkles, saved = None, [], 0
    for tick, truth in enumerate(path):
        frame = make_frame(truth, bgr)
        center = mw.track(frame, preset)
        canvas = mw.fade_canvas(canvas)
        if center is not None:
            if prev is not None:
                cv2.line(canvas, prev, center, mw.rainbow_color(tick),
                         mw.TRAIL_THICKNESS, cv2.LINE_AA)
            mw.spawn_sparkles(sparkles, center)
        prev = center
        out = mw.compose(frame, canvas, center, tick, preset, sparkles)
        if tick in (5, 12, 25):
            cv2.imwrite(os.path.join(OUT_DIR, f"verify_render_{tick:02d}.png"), out)
            saved += 1

    print(f"    프레임 {len(path)}개 렌더, 데모 이미지 {saved}장 저장")
    print("    → PASS (output/verify_render_*.png — 시각 확인용)")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 62)
    print(" 마법 지팡이 — 합성 데이터 검증 (웹캠 불필요)")
    print("=" * 62)
    test_centroid_accuracy()
    test_robustness()
    test_trail_matches_path()
    test_fade()
    test_glow()
    test_full_pipeline_render()
    print("\n" + "=" * 62)
    print(" 전체 통과. 단, 실제 웹캠 조명/모션블러 환경은 검증 대상 아님.")
    print("=" * 62)
