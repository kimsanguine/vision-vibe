"""문서 스캐너 — 종이 사진에서 4개 모서리를 찾아 원근 보정한다.

파이프라인:
    그레이스케일 → 블러 → Canny 엣지 → findContours
    → 면적 최대 4각형(approxPolyDP) → getPerspectiveTransform → warpPerspective

딥러닝 없이 OpenCV 고전 기법만 사용한다.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# 엣지 검출은 축소 이미지에서 수행한다 (속도 + 잡은 노이즈 억제).
# 찾은 꼭짓점은 원본 좌표로 되돌려 원본 해상도로 보정한다.
DETECT_WIDTH = 600

# 문서 후보 4각형이 전체 이미지에서 차지해야 하는 최소 면적 비율.
# 이보다 작으면 문서가 아니라 이미지 안의 작은 사각형(로고·표 등)으로 본다.
MIN_AREA_RATIO = 0.15


@dataclass
class ScanResult:
    """스캔 단계별 결과 (시각화 + 검증용)."""
    original: np.ndarray          # 원본 이미지
    edges: np.ndarray             # Canny 엣지 맵 (원본 크기로 확대)
    quad: np.ndarray              # 검출된 문서 4각형 (원본 좌표, TL·TR·BR·BL 순)
    warped: np.ndarray            # 원근 보정 결과
    used_fallback: bool           # 4각형 검출 실패로 전체 이미지를 쓴 경우 True
    note: str                     # 사람이 읽을 상태 메시지


def order_points(pts: np.ndarray) -> np.ndarray:
    """4개 점을 좌상단→우상단→우하단→좌하단 순으로 정렬한다.

    findContours가 주는 점 순서는 보장되지 않으므로,
    정렬 없이 getPerspectiveTransform에 넣으면 결과가 뒤집히거나 대각으로 꼬인다.

    판별 근거:
        x+y 최소 = 좌상단, 최대 = 우하단
        y-x 최소 = 우상단, 최대 = 좌하단
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR

    diff = np.diff(pts, axis=1).ravel()   # y - x
    rect[1] = pts[np.argmin(diff)]        # TR
    rect[3] = pts[np.argmax(diff)]        # BL

    # 같은 점이 두 번 배정되면 정렬이 실패한 것 — 조용히 넘기지 않는다.
    if len(np.unique(rect, axis=0)) != 4:
        raise ValueError(
            "꼭짓점 정렬 실패: 4각형이 지나치게 회전(45° 이상)했거나 뒤틀렸습니다."
        )
    return rect


def detect_edges(image: np.ndarray) -> np.ndarray:
    """그레이스케일 → 가우시안 블러 → Canny 엣지 검출."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 조명·대비가 이미지마다 달라 고정 임계값은 잘 깨진다.
    # 밝기 중앙값 기준 자동 임계값(auto-Canny)으로 잡는다.
    median = float(np.median(blurred))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(blurred, lower, upper)

    # 종이 경계가 조명 때문에 끊기면 contour가 닫히지 않아 4각형을 못 찾는다.
    # 살짝 팽창시켜 틈을 메운다.
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def find_document_quad(edges: np.ndarray, image_area: float) -> np.ndarray | None:
    """엣지 맵에서 문서로 볼 만한 가장 큰 4각형을 찾는다. 없으면 None."""
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        peri = cv2.arcLength(contour, closed=True)
        # epsilon = 둘레의 2% — 종이의 미세한 굴곡은 무시하고 직선으로 근사
        approx = cv2.approxPolyDP(contour, 0.02 * peri, closed=True)

        if len(approx) != 4:
            continue
        if cv2.contourArea(approx) < MIN_AREA_RATIO * image_area:
            continue
        if not cv2.isContourConvex(approx):
            continue
        return approx

    return None


def estimate_aspect_ratio(quad: np.ndarray, image_shape: tuple) -> float | None:
    """원근 투영된 4각형에서 원래 직사각형의 종횡비(가로/세로)를 복원한다.

    '긴 변을 쓴다'는 흔한 휴리스틱은 원근이 강하면 크게 틀린다.
    카메라에 가까운 변이 실제보다 길게 찍히기 때문이다.

    대신 소실점을 쓴다. 마주보는 변 두 쌍을 각각 연장해 만나는 두 소실점은
    3D 공간에서 서로 직교하는 두 방향을 가리킨다. 이 직교 조건이
    카메라 초점거리 f에 대한 방정식을 하나 주고, f를 알면 종횡비가 나온다.
    (Zhang & He, "Whiteboard scanning and image enhancement", 2006)

    추정이 불안정한 경우(변이 거의 평행 = 정면 촬영, f² ≤ 0 등)에는
    None을 반환해 호출측이 기존 휴리스틱을 쓰도록 한다.
    """
    h, w = image_shape[:2]
    u0, v0 = w / 2.0, h / 2.0          # 주점 = 이미지 중심으로 가정
    diagonal = float(np.hypot(w, h))

    tl, tr, br, bl = [np.array([p[0], p[1], 1.0]) for p in quad]

    # 가로 방향 두 변(상·하)의 교점, 세로 방향 두 변(좌·우)의 교점
    vp_h = np.cross(np.cross(tl, tr), np.cross(bl, br))
    vp_v = np.cross(np.cross(tl, bl), np.cross(tr, br))

    points = []
    for vp in (vp_h, vp_v):
        if abs(vp[2]) < 1e-9:
            return None                # 완전 평행 → 소실점이 무한대
        p = vp[:2] / vp[2]
        # 소실점이 지나치게 멀면 좌표 오차가 증폭돼 f 추정이 신뢰할 수 없다
        if np.hypot(p[0] - u0, p[1] - v0) > 50 * diagonal:
            return None
        points.append(p)

    p_h, p_v = points
    # 두 방향의 직교 조건에서 유도: f² = -[(x1-u0)(x2-u0) + (y1-v0)(y2-v0)]
    f_squared = -((p_h[0] - u0) * (p_v[0] - u0) + (p_h[1] - v0) * (p_v[1] - v0))
    if f_squared <= 0:
        return None                    # 물리적으로 불가능한 해 → 추정 포기

    f = np.sqrt(f_squared)
    k_inv = np.array([[1 / f, 0, -u0 / f], [0, 1 / f, -v0 / f], [0, 0, 1]])

    # 단위 정사각형 → 검출 4각형 호모그래피의 두 열이 가로·세로 축에 대응한다
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(unit, quad.astype(np.float32))
    axis_w = k_inv @ homography[:, 0]
    axis_h = k_inv @ homography[:, 1]

    denominator = np.linalg.norm(axis_h)
    if denominator < 1e-9:
        return None
    aspect = float(np.linalg.norm(axis_w) / denominator)

    # 종이로 볼 수 없는 극단값은 추정 실패로 간주
    return aspect if 0.05 < aspect < 20.0 else None


def four_point_transform(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """정렬된 4각형을 직사각형으로 펴서 반환한다."""
    tl, tr, br, bl = quad

    # 마주보는 두 변 중 긴 쪽을 출력 크기로 잡는다 (원근으로 짧아진 쪽 기준이면 찌그러짐)
    width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))

    # 소실점으로 진짜 종횡비를 복원할 수 있으면 그 비율을 따른다.
    # 출력 픽셀 수는 위 휴리스틱 크기와 비슷하게 유지한다.
    aspect = estimate_aspect_ratio(quad, image.shape)
    if aspect is not None:
        area = width * height
        height = np.sqrt(area / aspect)
        width = aspect * height

    out_w, out_h = max(int(round(width)), 1), max(int(round(height)), 1)

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image, matrix, (out_w, out_h))


def scan(image: np.ndarray, strict: bool = False) -> ScanResult:
    """문서 스캔 파이프라인 전체를 실행한다.

    strict=True 이면 4각형 검출 실패 시 예외를 던진다.
    strict=False(기본) 이면 이미지 전체를 문서로 간주하는 폴백을 쓰되,
    ScanResult.used_fallback=True 로 명시한다.
    """
    h, w = image.shape[:2]
    scale = DETECT_WIDTH / w
    small = cv2.resize(image, (DETECT_WIDTH, max(int(round(h * scale)), 1)))

    edges_small = detect_edges(small)
    quad_small = find_document_quad(edges_small, float(small.shape[0] * small.shape[1]))

    edges = cv2.resize(edges_small, (w, h), interpolation=cv2.INTER_NEAREST)

    if quad_small is None:
        message = (
            "문서 4각형을 찾지 못했습니다. "
            "(종이 경계가 배경과 충분히 대비되지 않거나, 모서리가 화면 밖으로 잘렸을 수 있습니다)"
        )
        if strict:
            raise ValueError(message)
        # 폴백: 이미지 전체를 문서로 간주 — 보정 없이 원본을 그대로 돌려준다
        full = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        return ScanResult(
            original=image,
            edges=edges,
            quad=full,
            warped=image.copy(),
            used_fallback=True,
            note=f"{message} → 폴백: 원근 보정 없이 원본 전체를 출력합니다.",
        )

    # 축소 좌표에서 찾은 꼭짓점을 원본 해상도로 되돌린다
    quad = order_points(quad_small.astype(np.float32) / scale)
    warped = four_point_transform(image, quad)

    return ScanResult(
        original=image,
        edges=edges,
        quad=quad,
        warped=warped,
        used_fallback=False,
        note=f"문서 4각형 검출 성공 → 보정 결과 {warped.shape[1]}x{warped.shape[0]}px",
    )


def make_visualization(result: ScanResult) -> np.ndarray:
    """원본(꼭짓점 표시) / 엣지 / 보정결과를 가로로 나란히 붙인 이미지를 만든다."""
    panel_h = 600

    def fit(img: np.ndarray, label: str) -> np.ndarray:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
        scale = panel_h / bgr.shape[0]
        bgr = cv2.resize(bgr, (max(int(round(bgr.shape[1] * scale)), 1), panel_h))
        # 라벨 가독성을 위해 검은 배경 띠를 깔고 흰 글씨
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(bgr, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return bgr

    annotated = result.original.copy()
    if not result.used_fallback:
        cv2.drawContours(annotated, [result.quad.astype(int)], -1, (0, 255, 0), 3)
        for i, (x, y) in enumerate(result.quad.astype(int)):
            cv2.circle(annotated, (int(x), int(y)), 10, (0, 0, 255), -1)
            cv2.putText(annotated, "TL TR BR BL".split()[i], (int(x) + 14, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    warped_label = "3. FALLBACK (보정 안 함)" if result.used_fallback else "3. warpPerspective"
    panels = [
        fit(annotated, "1. original + corners"),
        fit(result.edges, "2. Canny edges"),
        fit(result.warped, warped_label),
    ]
    return np.hstack(panels)


def main() -> int:
    parser = argparse.ArgumentParser(description="종이 사진을 반듯하게 펴주는 문서 스캐너")
    parser.add_argument("input", help="입력 이미지 경로")
    parser.add_argument("-o", "--output", help="보정 결과 저장 경로 (기본: output/<이름>_scanned.jpg)")
    parser.add_argument("--viz", help="단계별 시각화 저장 경로 (기본: output/<이름>_viz.jpg)")
    parser.add_argument("--strict", action="store_true",
                        help="4각형 검출 실패 시 폴백 없이 에러로 종료")
    args = parser.parse_args()

    src = Path(args.input)
    image = cv2.imread(str(src))
    if image is None:
        print(f"[에러] 이미지를 읽을 수 없습니다: {src}", file=sys.stderr)
        return 1

    try:
        result = scan(image, strict=args.strict)
    except ValueError as e:
        print(f"[에러] {e}", file=sys.stderr)
        return 2

    out_dir = src.parent.parent / "output" if src.parent.name == "samples" else Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / f"{src.stem}_scanned.jpg"
    viz_path = Path(args.viz) if args.viz else out_dir / f"{src.stem}_viz.jpg"

    cv2.imwrite(str(out_path), result.warped)
    cv2.imwrite(str(viz_path), make_visualization(result))

    prefix = "[경고]" if result.used_fallback else "[완료]"
    print(f"{prefix} {result.note}")
    print(f"  보정 결과 : {out_path}")
    print(f"  시각화    : {viz_path}")
    return 3 if result.used_fallback else 0


if __name__ == "__main__":
    sys.exit(main())
