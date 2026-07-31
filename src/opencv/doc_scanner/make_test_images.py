"""검증용 합성 문서 사진을 만든다 (실제 사진이 없으므로).

PIL로 '종이'(흰 배경 + 검은 텍스트 줄)를 그린 뒤,
카메라 기울임을 3D로 시뮬레이션해 배경 위에 원근 투영으로 합성한다.

임의로 사다리꼴 네 점을 찍지 않고 실제 투영을 계산하는 이유:
그래야 '원래 종횡비'가 정의되고, 보정 결과를 수치로 채점할 수 있다.
각 이미지 옆에 정답 꼭짓점 좌표를 JSON으로 함께 저장한다.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# A4 비율(210x297mm ≈ 0.707)에 가까운 종이
PAGE_W, PAGE_H = 620, 877
SCENE_W, SCENE_H = 1000, 1300

FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",   # 한글 글리프 포함
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _load_font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return None


def make_page() -> np.ndarray:
    """흰 종이에 제목 + 텍스트 줄을 그린 이미지를 BGR ndarray로 반환한다."""
    page = Image.new("RGB", (PAGE_W, PAGE_H), (252, 251, 248))
    draw = ImageDraw.Draw(page)

    title_font = _load_font(46)
    body_font = _load_font(24)

    margin = 60
    y = 90
    if title_font:
        draw.text((margin, y), "RECEIPT / 영수증", fill=(20, 20, 20), font=title_font)
    else:
        draw.rectangle([margin, y, margin + 380, y + 40], fill=(20, 20, 20))
    y += 90
    draw.line([(margin, y), (PAGE_W - margin, y)], fill=(40, 40, 40), width=3)
    y += 40

    # 본문 텍스트 줄 — 폰트가 없으면 검은 막대로 대체
    for i in range(16):
        line_w = int((PAGE_W - 2 * margin) * (0.95 if i % 4 else 0.6))
        if body_font:
            draw.text((margin, y), "OpenCV document scanner test line %02d" % i,
                      fill=(35, 35, 35), font=body_font)
        else:
            draw.rectangle([margin, y, margin + line_w, y + 14], fill=(35, 35, 35))
        y += 38

    return cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)


def make_background(rng: np.random.Generator) -> np.ndarray:
    """책상 느낌의 배경 — 밝기 그라데이션 + 노이즈."""
    grad_y = np.linspace(70, 115, SCENE_H, dtype=np.float32)[:, None]
    grad_x = np.linspace(-12, 12, SCENE_W, dtype=np.float32)[None, :]
    base = grad_y + grad_x
    noise = rng.normal(0, 5.0, (SCENE_H, SCENE_W)).astype(np.float32)
    gray = np.clip(base + noise, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.int16)
    bgr[:, :, 0] = np.clip(bgr[:, :, 0] + 14, 0, 255)   # 살짝 푸른 책상 톤
    return bgr.astype(np.uint8)


def project_page(tilt_deg: float, pan_deg: float, roll_deg: float,
                 distance: float, shift: tuple[float, float]) -> np.ndarray:
    """종이 네 모서리를 기울인 카메라로 투영한 좌표를 계산한다.

    종이를 z=0 평면의 직사각형으로 두고,
    roll(광축 회전) → tilt(x축) → pan(y축) 순으로 돌린 뒤 핀홀 투영한다.
    임의 사다리꼴이 아니라 실제 투영이므로 '정답 종횡비'가 정의된다.

    중요: 주점(principal point)은 반드시 이미지 중심에 둔다.
    투영 후 2D 확대·평행이동으로 화면을 맞추면 주점이 중심에서 벗어나
    실제 카메라와 다른 이미지가 되고, 소실점 기반 검증이 왜곡된다.
    초점거리는 종이가 여백 안에 들어오도록 자동으로 정한다
    (초점거리 조정 = 주점 기준 확대이므로 주점은 중심에 유지된다).
    """
    hw, hh = PAGE_W / 2, PAGE_H / 2
    corners = np.array([[-hw, -hh, 0], [hw, -hh, 0], [hw, hh, 0], [-hw, hh, 0]],
                       dtype=np.float64)  # TL, TR, BR, BL (종이 기준)

    def rot_x(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def rot_y(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rot_z(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    r = rot_y(np.radians(pan_deg)) @ rot_x(np.radians(tilt_deg)) @ rot_z(np.radians(roll_deg))
    pts = corners @ r.T
    pts[:, 0] += shift[0]
    pts[:, 1] += shift[1]
    pts[:, 2] += distance

    normalized = np.column_stack([pts[:, 0] / pts[:, 2], pts[:, 1] / pts[:, 2]])

    # 초점거리 f를 키우면 주점 기준으로 확대된다 → 여백에 딱 맞는 f를 고른다
    margin = 70
    limit_x = (SCENE_W / 2 - margin) / np.abs(normalized[:, 0]).max()
    limit_y = (SCENE_H / 2 - margin) / np.abs(normalized[:, 1]).max()
    focal = min(limit_x, limit_y)

    center = np.array([SCENE_W / 2, SCENE_H / 2])
    return (center + focal * normalized).astype(np.float32)


def compose(page: np.ndarray, quad: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """종이를 배경 위 지정된 4각형 위치에 원근 합성하고 그림자를 넣는다."""
    src = np.array([[0, 0], [PAGE_W - 1, 0], [PAGE_W - 1, PAGE_H - 1], [0, PAGE_H - 1]],
                   dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, quad)

    warped_page = cv2.warpPerspective(page, matrix, (SCENE_W, SCENE_H))
    mask = cv2.warpPerspective(np.full((PAGE_H, PAGE_W), 255, np.uint8), matrix,
                               (SCENE_W, SCENE_H))

    scene = make_background(rng)

    # 종이 아래 그림자 — 종이/배경 경계가 무조건 선명하지만은 않게 만든다
    shadow = cv2.GaussianBlur(mask, (61, 61), 0).astype(np.float32) / 255.0
    offset = np.float32([[1, 0, 12], [0, 1, 16]])
    shadow = cv2.warpAffine(shadow, offset, (SCENE_W, SCENE_H))
    scene = (scene.astype(np.float32) * (1 - 0.45 * shadow[:, :, None])).astype(np.uint8)

    alpha = (cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0)[:, :, None]
    out = (warped_page.astype(np.float32) * alpha + scene.astype(np.float32) * (1 - alpha))
    return np.clip(out, 0, 255).astype(np.uint8)


CASES = {
    # (a) 살짝 기울어짐 — 거의 정면, 광축 회전 8°
    "case_a_slight_tilt": dict(tilt_deg=6.0, pan_deg=3.0, roll_deg=8.0,
                               distance=3000.0, shift=(30.0, 0.0)),
    # (b) 원근이 심한 사다리꼴 — 크게 기울여 위/아래 폭 차이를 크게
    "case_b_strong_perspective": dict(tilt_deg=48.0, pan_deg=16.0, roll_deg=-5.0,
                                      distance=1250.0, shift=(-60.0, 40.0)),
}

# 배경 노이즈 시드. hash()는 프로세스마다 값이 달라져 검증 수치가 흔들리므로 고정한다.
SEEDS = {"case_a_slight_tilt": 101, "case_b_strong_perspective": 202}


def main() -> None:
    out_dir = Path(__file__).parent / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    page = make_page()
    manifest = {}

    for name, params in CASES.items():
        rng = np.random.default_rng(SEEDS[name])
        quad = project_page(**params)
        scene = compose(page, quad, rng)

        img_path = out_dir / f"{name}.jpg"
        cv2.imwrite(str(img_path), scene, [cv2.IMWRITE_JPEG_QUALITY, 92])

        top = float(np.linalg.norm(quad[1] - quad[0]))
        bottom = float(np.linalg.norm(quad[2] - quad[3]))
        manifest[name] = {
            "image": img_path.name,
            "true_page_size": [PAGE_W, PAGE_H],
            "true_aspect_w_over_h": PAGE_W / PAGE_H,
            "gt_corners_tl_tr_br_bl": quad.tolist(),
            "top_over_bottom_edge_ratio": top / bottom,  # 1에 가까울수록 원근 약함
            "params": params,
        }
        print(f"생성: {img_path.name}  상단/하단 변 비율={top / bottom:.3f}")

    # 폴백 경로 검증용 — 문서가 없는 배경. 정답이 없으므로 manifest에는 넣지 않는다.
    rng = np.random.default_rng(7)
    no_doc = make_background(rng)
    cv2.imwrite(str(out_dir / "case_c_no_document.jpg"), no_doc,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("생성: case_c_no_document.jpg  (문서 없음 — 폴백/에러 경로 확인용)")

    manifest_path = out_dir / "ground_truth.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"정답 좌표: {manifest_path}")


if __name__ == "__main__":
    main()
