"""합성 이미지로 스캐너를 채점한다.

두 가지를 수치로 잰다.
    1) 꼭짓점 오차 — 검출한 4점이 정답 4점에서 몇 px 벗어났는가 (대각선 대비 %)
    2) 종횡비 오차 — 보정 결과의 w/h가 원래 종이의 w/h에서 몇 % 벗어났는가

주의: 이 채점은 합성 이미지에 대한 것이다. 실제 스마트폰 사진의 성능을 뜻하지 않는다.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from scan import order_points, scan

# 합격선 — 합성 이미지 기준의 자체 기준선이며 업계 표준이 아니다
CORNER_TOL_PCT = 1.0    # 이미지 대각선의 1% 이내
ASPECT_TOL_PCT = 10.0   # 종횡비 상대오차 10% 이내


def evaluate(case_name: str, meta: dict, samples_dir: Path) -> dict:
    image = cv2.imread(str(samples_dir / meta["image"]))
    if image is None:
        raise FileNotFoundError(samples_dir / meta["image"])

    result = scan(image)

    gt = order_points(np.array(meta["gt_corners_tl_tr_br_bl"], dtype=np.float32))
    per_corner = np.linalg.norm(result.quad - gt, axis=1)
    diagonal = float(np.hypot(*image.shape[:2]))

    true_aspect = meta["true_aspect_w_over_h"]
    got_aspect = result.warped.shape[1] / result.warped.shape[0]

    # 비교용 baseline: 소실점 보정 없이 '긴 변을 쓴다'는 흔한 휴리스틱만 적용한 경우
    tl, tr, br, bl = result.quad
    base_w = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    base_h = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    base_aspect = float(base_w / base_h)

    corner_pct = float(per_corner.mean()) / diagonal * 100
    aspect_pct = abs(got_aspect - true_aspect) / true_aspect * 100

    return {
        "case": case_name,
        "detected": not result.used_fallback,
        "corner_err_px_mean": float(per_corner.mean()),
        "corner_err_px_max": float(per_corner.max()),
        "corner_err_pct_of_diagonal": corner_pct,
        "true_aspect": true_aspect,
        "recovered_aspect": got_aspect,
        "aspect_err_pct": aspect_pct,
        "baseline_maxedge_aspect": base_aspect,
        "baseline_maxedge_err_pct": abs(base_aspect - true_aspect) / true_aspect * 100,
        "warped_size": [result.warped.shape[1], result.warped.shape[0]],
        "pass": (not result.used_fallback
                 and corner_pct <= CORNER_TOL_PCT
                 and aspect_pct <= ASPECT_TOL_PCT),
    }


def main() -> int:
    samples_dir = Path(__file__).parent / "samples"
    manifest_path = samples_dir / "ground_truth.json"
    if not manifest_path.exists():
        print("먼저 `python make_test_images.py`로 합성 이미지를 만드세요.", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [evaluate(name, meta, samples_dir) for name, meta in manifest.items()]

    print(f"{'케이스':<28} {'검출':<5} {'꼭짓점오차(px)':>14} {'대각선대비':>10} "
          f"{'정답비율':>9} {'복원비율':>9} {'비율오차':>9} {'휴리스틱만':>11}  판정")
    print("-" * 118)
    for r in rows:
        print(f"{r['case']:<28} {'O' if r['detected'] else 'X':<5} "
              f"{r['corner_err_px_mean']:>14.2f} {r['corner_err_pct_of_diagonal']:>9.3f}% "
              f"{r['true_aspect']:>9.4f} {r['recovered_aspect']:>9.4f} "
              f"{r['aspect_err_pct']:>8.2f}% {r['baseline_maxedge_err_pct']:>10.2f}%  "
              f"{'PASS' if r['pass'] else 'FAIL'}")

    print(f"\n합격 기준: 꼭짓점 오차 ≤ 대각선의 {CORNER_TOL_PCT}%, "
          f"종횡비 오차 ≤ {ASPECT_TOL_PCT}%")
    print("'휴리스틱만' = 소실점 보정 없이 '긴 변 채택'만 썼을 때의 종횡비 오차 (비교용)")
    print("주의: 합성 이미지 기준 수치입니다. 실제 스마트폰 사진 성능이 아닙니다.")

    (Path(__file__).parent / "output" / "verify_report.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0 if all(r["pass"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
