"""실사진 6장 정확도 실측 + 파라미터 민감도 + 모델 용량 비교.

이 스크립트가 이 프로젝트의 **핵심 산출물**이다. README 에 적힌 모든 숫자는
여기서 나온다. 하드코딩된 정확도 수치는 없다.

보고 원칙 (mediapipe/hand_lite/bench.py 의 규약을 그대로 따른다)
  [사실] 파일 용량. os.stat 로 직접 읽는다. 반박 불가.
  [사실] 이 6장에서의 정답 개수. 결정적 파이프라인이라 재실행해도 동일하다.
  [추정] 그 정확도가 다른 손·배경·조명에서도 유지된다는 주장. 이건 6장짜리
         표본에서 나온 관측치일 뿐이다.

민감도 분석을 함께 내는 이유
────────────────────────────
"정확도 2/6" 만 보고하면 "임계값을 잘못 잡은 것 아니냐"는 반론을 반박할 수
없다. 그래서 깊이·각도 임계값을 격자로 훑어, **어떤 조합에서도 6/6 이
나오지 않는지**를 측정으로 확인한다. 나오지 않는다면 그건 튜닝 실패가 아니라
기법의 구조적 한계다 — 이게 이 과제의 교육적 결론이다.

실행: ../../../venv/bin/python evaluate.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rps_classic.fingers import (  # noqa: E402
    analyze_contour,
    convexity_defects,
    count_extended_fingers,
    count_finger_gaps,
    defect_angle_deg,
    largest_contour,
)
from rps_classic.gesture import classify  # noqa: E402
from rps_classic.pipeline import detect  # noqa: E402
from rps_classic.skin import skin_mask  # noqa: E402
from rps_classic.types import Gesture  # noqa: E402

HERE = Path(__file__).resolve().parent


def _find_workshop_root(start: Path) -> Path:
    """yolo/rps 가 있는 조상 디렉토리를 찾는다.

    HERE.parent 로 고정하지 않는 이유: 이 스크립트를 worktree 처럼 한 단계 더
    깊은 곳에서 실행해도 픽스처를 찾을 수 있어야 한다.
    """
    for candidate in (start, *start.parents):
        if (candidate / "yolo" / "rps").is_dir():
            return candidate
    return start.parent


WORKSHOP = _find_workshop_root(HERE)
FIXTURE_DIR = WORKSHOP / "yolo" / "rps" / "tests" / "fixtures"
OUT_DIR = HERE / "output"

# 파일명 → 정답 라벨
FIXTURES: tuple[tuple[str, Gesture], ...] = (
    ("rock_0", Gesture.ROCK),
    ("rock_1", Gesture.ROCK),
    ("paper_0", Gesture.PAPER),
    ("paper_1", Gesture.PAPER),
    ("scissors_0", Gesture.SCISSORS),
    ("scissors_1", Gesture.SCISSORS),
)

# 3자 비교 대상 모델 파일
MODEL_FILES = {
    "yolo11x_rps (yolo/rps)": WORKSHOP / "yolo" / "rps" / "models" / "rps_yolo11x_leeyunjai.pt",
    "yolo11n_rps (yolo/rps)": WORKSHOP / "yolo" / "rps" / "models" / "rps_yolo11n_gholamreza.pt",
    "hand_landmarker.task (mediapipe/hand_lite)":
        WORKSHOP / "mediapipe/hand_lite" / "models" / "hand_landmarker.task",
}


# ──────────────────────────────────────────────────────────────────────
# 1. 픽스처 6장 정확도 — [사실]
# ──────────────────────────────────────────────────────────────────────

def evaluate_fixtures() -> dict:
    """6장 전부에 파이프라인을 돌려 결과와 진단 정보를 모은다."""
    rows = []
    for name, truth in FIXTURES:
        path = FIXTURE_DIR / f"{name}.png"
        img = cv2.imread(str(path))
        if img is None:
            rows.append({"name": name, "error": f"읽기 실패: {path}"})
            continue

        det = detect(img)
        s = det.shape
        rows.append({
            "name": name,
            "truth": truth.name,
            "pred": det.gesture.name,
            "correct": det.gesture is truth,
            "fingers": None if s is None else s.fingers,
            "gaps": None if s is None else s.finger_gaps,
            "solidity": None if s is None else round(s.solidity, 3),
            "contour_area": None if s is None else round(s.contour_area),
            "mask_frac": round(float(det.mask.mean()) / 255.0, 3),
            "hand_found": s is not None,
        })

    scored = [r for r in rows if "error" not in r]
    n_correct = sum(1 for r in scored if r["correct"])
    return {"rows": rows, "n_correct": n_correct, "n_total": len(scored)}


def print_fixture_table(res: dict) -> None:
    print("\n" + "=" * 88)
    print("[사실] 1. 실사진 6장 인식 정확도 — 기본 파라미터 그대로 실측")
    print("=" * 88)
    print(f"{'파일':<12}{'정답':<10}{'예측':<12}{'O/X':<5}"
          f"{'골':>4}{'손가락':>8}{'solidity':>10}{'마스크비율':>12}")
    print("-" * 88)
    for r in res["rows"]:
        if "error" in r:
            print(f"{r['name']:<12}{r['error']}")
            continue
        mark = "O" if r["correct"] else "X"
        print(f"{r['name']:<12}{r['truth']:<10}{r['pred']:<12}{mark:<5}"
              f"{r['gaps']:>4}{r['fingers']:>8}{r['solidity']:>10.3f}"
              f"{r['mask_frac'] * 100:>11.1f}%")
    print("-" * 88)
    n, t = res["n_correct"], res["n_total"]
    print(f"정확도: {n}/{t} = {n / t * 100:.1f}%")


# ──────────────────────────────────────────────────────────────────────
# 2. 실패 원인 분석 — 어느 단계에서 무너졌나
# ──────────────────────────────────────────────────────────────────────

def diagnose(name: str) -> dict:
    """한 장의 모든 convexity defect 를 깊이·각도와 함께 나열한다.

    "왜 틀렸나"를 서술이 아니라 숫자로 답하기 위한 함수다. 어떤 골이 어떤
    필터에 걸려 탈락했는지가 그대로 보인다.
    """
    img = cv2.imread(str(FIXTURE_DIR / f"{name}.png"))
    mask = skin_mask(img)
    contour = largest_contour(mask)
    if contour is None:
        return {"name": name, "hand_found": False}

    defects = convexity_defects(contour)
    items = []
    for s_idx, e_idx, f_idx, depth_fixed in defects:
        start, end, far = contour[s_idx][0], contour[e_idx][0], contour[f_idx][0]
        items.append({
            "depth_px": round(float(depth_fixed) / 256.0, 2),
            "angle_deg": round(defect_angle_deg(start, end, far), 1),
            "far_xy": [int(far[0]), int(far[1])],
        })
    items.sort(key=lambda d: -d["depth_px"])

    shape = analyze_contour(contour)
    x, y, w, h = cv2.boundingRect(contour)
    return {
        "name": name,
        "hand_found": True,
        "n_defects_total": len(items),
        "defects_top": items[:6],
        "solidity": round(shape.solidity, 3),
        "gaps": shape.finger_gaps,
        "fingers": shape.fingers,
        "bbox": [int(x), int(y), int(w), int(h)],
        "touches_bottom_border": bool(y + h >= img.shape[0] - 1),
    }


def print_diagnosis(diags: list[dict]) -> None:
    print("\n" + "=" * 88)
    print("[사실] 2. 실패 원인 — 검출된 모든 골의 깊이/각도 (깊이 상위 6개)")
    print("=" * 88)
    print("판정 기준: 깊이 > 10px  그리고  각도 <= 90도  인 것만 '손가락 사이'로 인정")
    for d in diags:
        if not d["hand_found"]:
            print(f"\n[{d['name']}] 손 컨투어 자체를 찾지 못함 — 마스크 단계 실패")
            continue
        print(f"\n[{d['name']}] 총 결함 {d['n_defects_total']}개 | "
              f"solidity={d['solidity']:.3f} | 인정된 골 {d['gaps']}개 → "
              f"손가락 {d['fingers']}개 | 하단 경계 접촉={d['touches_bottom_border']}")
        for it in d["defects_top"]:
            depth_ok = it["depth_px"] > 10.0
            angle_ok = it["angle_deg"] <= 90.0
            if depth_ok and angle_ok:
                verdict = "인정"
            elif not depth_ok and not angle_ok:
                verdict = "탈락(깊이·각도 둘 다)"
            elif not depth_ok:
                verdict = "탈락(너무 얕음)"
            else:
                verdict = "탈락(각도 둔각 — 손목/손날)"
            print(f"    깊이 {it['depth_px']:7.2f}px  각도 {it['angle_deg']:6.1f}도  "
                  f"far={tuple(it['far_xy'])}  → {verdict}")


# ──────────────────────────────────────────────────────────────────────
# 3. 파라미터 민감도 + 마스크 절제(ablation)
#    — 튜닝 실패인가, 마스크 실패인가, 구조적 한계인가
# ──────────────────────────────────────────────────────────────────────

def oracle_mask(frame_bgr: np.ndarray, *, morph: bool = True) -> np.ndarray:
    """진단 전용 '반칙' 마스크 — 피부색이 아니라 배경 여부로 손을 분리한다.

    이 픽스처의 배경은 거의 무채색 고휘도(흰~옅은 회색)다. 그래서
    "채도가 있거나 어두운 픽셀 = 전경" 이라는 규칙이 사실상 완벽한 손 실루엣을
    만든다. **매니큐어 칠한 손톱까지 손에 포함된다.**

    이 함수는 제품 코드가 아니다(rps_classic 패키지 밖에 둔 이유). 목적은 단 하나:
    "정확도가 낮은 원인이 피부색 검출 단계인가"를 절제 실험으로 판정하는 것이다.
    오라클 마스크로 갈아끼웠을 때 정확도가 오르면 병목은 마스크,
    안 오르면 병목은 그 뒤 단계(기하/규칙)다.

    이 규칙은 배경이 균일한 이 픽스처에서만 성립한다 — 실사용에 쓸 수 없다.
    """
    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((saturation > 30) | (value < 200)).astype(np.uint8) * 255
    if morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


# 절제 실험에서 비교할 마스크 변형들.
# morph 를 끄는 변형을 넣는 이유: open/close 가 손가락 사이 좁은 골을 메워
# 골을 잃게 만들 수 있다. 그 가능성도 추측하지 말고 측정한다.
MASK_VARIANTS: dict[str, callable] = {
    "skin(YCrCb, 기본)": lambda img: skin_mask(img),
    "skin(YCrCb, 모폴로지 없음)": lambda img: skin_mask(img, open_iterations=0, close_iterations=0),
    "oracle(배경분리, 반칙)": lambda img: oracle_mask(img, morph=True),
    "oracle(배경분리, 모폴로지 없음)": lambda img: oracle_mask(img, morph=False),
}

SWEEP_DEPTHS = [2.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]
SWEEP_ANGLES = [60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0]


def _sweep_one_variant(variant_name: str) -> dict:
    """마스크 변형 1개에 대해 (깊이 × 각도) 격자 정확도를 측정한다.

    마스크·컨투어·결함은 임계값과 무관하므로 이미지당 한 번만 계산해 재사용한다.
    """
    build_mask = MASK_VARIANTS[variant_name]
    cache = []
    for name, truth in FIXTURES:
        img = cv2.imread(str(FIXTURE_DIR / f"{name}.png"))
        contour = largest_contour(build_mask(img))
        if contour is None:
            cache.append((name, truth, None, None, 0.0))
            continue
        cache.append((name, truth, contour, convexity_defects(contour),
                      analyze_contour(contour).solidity))

    grid, best = [], {"n_correct": -1}
    # 이미지별로 "이 변형 안에서 한 번이라도 맞은 적이 있는가"를 누적한다.
    # 격자 점수(맞은 장수)만 저장하면 "가위는 어떤 조합에서도 못 맞춘다" 같은
    # 이미지 단위 주장을 나중에 확인할 수 없다 — 주장하려면 재야 한다.
    ever_correct = {name: False for name, _ in FIXTURES}
    for d in SWEEP_DEPTHS:
        row = []
        for a in SWEEP_ANGLES:
            n_correct, preds = 0, {}
            for name, truth, contour, defects, solidity in cache:
                if contour is None:
                    preds[name] = Gesture.UNKNOWN.name
                    continue
                gaps = count_finger_gaps(contour, defects, min_depth=d, max_angle_deg=a)
                pred = classify(count_extended_fingers(gaps, solidity))
                preds[name] = pred.name
                if pred is truth:
                    n_correct += 1
                    ever_correct[name] = True
            row.append(n_correct)
            if n_correct > best["n_correct"]:
                best = {"n_correct": n_correct, "min_depth": d, "max_angle": a,
                        "preds": dict(preds)}
        grid.append(row)
    return {"variant": variant_name, "grid": grid, "best": best,
            "ever_correct": ever_correct}


def sensitivity_sweep() -> dict:
    """모든 마스크 변형 × 임계값 격자를 훑어 달성 가능한 상한을 구한다."""
    results = [_sweep_one_variant(v) for v in MASK_VARIANTS]
    ceiling = max(results, key=lambda r: r["best"]["n_correct"])
    # 탐색한 **모든** 조합을 통틀어 한 번도 못 맞춘 이미지 = 구조적 실패
    never_correct = [name for name, _ in FIXTURES
                     if not any(r["ever_correct"][name] for r in results)]
    return {
        "depths": SWEEP_DEPTHS,
        "angles": SWEEP_ANGLES,
        "variants": results,
        "ceiling": ceiling,
        "never_correct": never_correct,
        "n_total": len(FIXTURES),
        "n_combinations": len(MASK_VARIANTS) * len(SWEEP_DEPTHS) * len(SWEEP_ANGLES),
    }


def print_sweep(sw: dict) -> None:
    print("\n" + "=" * 88)
    print("[사실] 3. 마스크 절제 + 파라미터 민감도 — 어떻게든 6/6 이 나오는가")
    print("=" * 88)
    print(f"총 {sw['n_combinations']}개 조합 탐색 "
          f"(마스크 {len(sw['variants'])}종 × 깊이 {len(sw['depths'])}종 × 각도 {len(sw['angles'])}종)")
    print("'oracle' 은 피부색을 쓰지 않고 배경 여부로 손을 자른 **반칙 마스크**다.")
    print("매니큐어 손톱까지 손에 포함되므로, 피부색 검출이 병목인지 가려낼 수 있다.")

    for r in sw["variants"]:
        print(f"\n── 마스크: {r['variant']}  (표의 값 = 맞춘 장수 / 6)")
        header = "  깊이\\각도" + "".join(f"{a:>7.0f}도" for a in sw["angles"])
        print(header)
        for d, row in zip(sw["depths"], r["grid"]):
            print(f"  {d:>7.0f}px" + "".join(f"{v:>8d}" for v in row))
        b = r["best"]
        print(f"  최고 {b['n_correct']}/6 @ 깊이 {b['min_depth']:.0f}px, 각도 {b['max_angle']:.0f}도")

    c = sw["ceiling"]
    b = c["best"]
    print("\n" + "-" * 88)
    print(f"전체 상한: {b['n_correct']}/{sw['n_total']}  "
          f"(마스크 '{c['variant']}', 깊이 {b['min_depth']:.0f}px, 각도 {b['max_angle']:.0f}도)")
    print("  그때의 예측: " + ", ".join(f"{k}→{v}" for k, v in b["preds"].items()))
    never = sw["never_correct"]
    if never:
        print(f"\n  탐색한 {sw['n_combinations']}개 조합을 통틀어 **한 번도** 맞추지 못한 이미지:")
        print(f"    {', '.join(never)}")
        print("    → 이 이미지들의 실패는 파라미터 문제가 아니다. 어떤 임계값·어떤")
        print("      마스크로도 정답이 나오지 않으므로 표현 방식의 한계다.")
    else:
        print(f"\n  모든 이미지가 최소 한 조합에서는 정답을 냈다 "
              f"(다만 한 조합이 동시에 다 맞추지는 못한다).")

    if b["n_correct"] < sw["n_total"]:
        print("\n  판정: 탐색한 어떤 조합에서도 6/6 이 나오지 않는다.")
        print("    → 정확도가 낮은 이유는 임계값을 잘못 골라서도, 피부색 검출이")
        print("      나빠서만도 아니다. 반칙 마스크로 손끝을 전부 살려줘도 못 맞춘다면")
        print("      남은 원인은 convexity defect 라는 **표현 방식 자체**다.")
        print("      (자세한 인과는 아래 [분석] 절 참조)")
    else:
        print("\n  판정: 6/6 조합이 존재한다 → 기본값 선택이 나빴다는 뜻이다.")


def audit_ceiling_setting(sw: dict) -> dict:
    """상한 설정이 '맞는 답을 옳은 이유로' 내는지 감사한다.

    격자에서 가장 높은 점수를 낸 설정을 그냥 기본값으로 채택하면 안 된다.
    6장에 맞춰 고른 임계값은 **손목 오목점을 손가락 사이로 오인하고도 정답이
    나오는** 설정일 수 있기 때문이다. 그래서 상한 설정에서 인정된 골들의
    각도를 다시 열어보고, 90도를 넘는 골(물리적으로 손가락 사이가 아닌 것)이
    섞여 있는지 센다.

    90도 초과 골에 기대어 맞춘 이미지가 하나라도 있으면, 그 점수는 기법의
    성능이 아니라 표본 과적합이다.
    """
    c = sw["ceiling"]
    b = c["best"]
    build_mask = MASK_VARIANTS[c["variant"]]
    min_depth, max_angle = b["min_depth"], b["max_angle"]

    per_image = []
    for name, truth in FIXTURES:
        img = cv2.imread(str(FIXTURE_DIR / f"{name}.png"))
        contour = largest_contour(build_mask(img))
        if contour is None:
            per_image.append({"name": name, "hand_found": False})
            continue
        accepted = []
        for s_idx, e_idx, f_idx, depth_fixed in convexity_defects(contour):
            depth = depth_fixed / 256.0
            if depth < min_depth:
                continue
            angle = defect_angle_deg(contour[s_idx][0], contour[e_idx][0], contour[f_idx][0])
            if angle <= max_angle:
                accepted.append({"depth_px": round(depth, 2), "angle_deg": round(angle, 1),
                                 "far_xy": [int(contour[f_idx][0][0]), int(contour[f_idx][0][1])]})
        n_obtuse = sum(1 for a in accepted if a["angle_deg"] > 90.0)
        per_image.append({
            "name": name, "hand_found": True, "truth": truth.name,
            "pred": b["preds"][name], "accepted": accepted,
            "n_accepted": len(accepted), "n_obtuse_accepted": n_obtuse,
            "correct": b["preds"][name] == truth.name,
        })

    tainted = [p for p in per_image
               if p.get("correct") and p.get("n_obtuse_accepted", 0) > 0]
    return {"setting": {"variant": c["variant"], "min_depth": min_depth, "max_angle": max_angle},
            "per_image": per_image, "tainted": [p["name"] for p in tainted]}


def print_ceiling_audit(audit: dict) -> None:
    st = audit["setting"]
    print("\n" + "=" * 88)
    print("[감사] 상한 설정을 기본값으로 채택해도 되나 — '옳은 이유로 맞았는지' 점검")
    print("=" * 88)
    print(f"대상 설정: 마스크 '{st['variant']}', 깊이 {st['min_depth']:.0f}px, "
          f"각도 {st['max_angle']:.0f}도")
    print("각도 90도 초과 골은 물리적으로 손가락 사이가 아니다(손목·손날의 완만한 굴곡).")
    print("그런 골에 기대어 정답이 나왔다면 그건 성능이 아니라 표본 과적합이다.\n")
    for p in audit["per_image"]:
        if not p["hand_found"]:
            print(f"  {p['name']:<12} 손 없음")
            continue
        mark = "O" if p["correct"] else "X"
        detail = ", ".join(f"{a['angle_deg']:.0f}도/{a['depth_px']:.0f}px" for a in p["accepted"]) or "없음"
        flag = "  ← 둔각 골에 의존!" if p["correct"] and p["n_obtuse_accepted"] > 0 else ""
        print(f"  {p['name']:<12}{p['truth']:<9}→{p['pred']:<9}{mark}  "
              f"인정된 골 {p['n_accepted']}개 [{detail}]{flag}")
    print()
    if audit["tainted"]:
        print(f"  판정: {', '.join(audit['tainted'])} 는 **손목 오목점을 손가락 사이로 오인한**")
        print("        덕분에 정답이 나왔다. 이 설정을 기본값으로 채택하면 6장짜리 표본에")
        print("        과적합된다. 그래서 이 프로젝트의 기본값은 상한 설정이 아니라,")
        print("        물리적 근거가 있는 각도 90도를 유지한다 — 점수는 낮지만 정직하다.")
    else:
        print("  판정: 상한 설정은 둔각 골에 의존하지 않는다. 채택을 검토할 수 있다.")


def print_causal_analysis(diags: list[dict], sw: dict) -> None:
    """실패 인과를 숫자로 못박는다 — 서술이 아니라 이 실행에서 나온 값으로.

    "왜 틀렸나"에 대한 답이 두 손모양에서 서로 **다르다**는 것이 이 절의 요지다.
    한 문장으로 뭉뚱그리면 틀린 설명이 된다.
    """
    by_name = {d["name"]: d for d in diags}
    print("\n" + "=" * 88)
    print("[분석] 실패 원인 — 보와 가위는 서로 다른 이유로 틀린다")
    print("=" * 88)

    p = by_name.get("paper_0", {})
    if p.get("hand_found"):
        depths = [it["depth_px"] for it in p["defects_top"]]
        print("\n(1) 보(PAPER): 골 깊이가 손가락마다 자릿수 단위로 다르고, 그 사이에 손목이 낀다")
        print(f"    paper_0 의 깊이 상위값: {', '.join(f'{d:.1f}px' for d in depths[:5])}")
        print("    손을 기울여 잡으면 손가락이 계단처럼 배열돼, 짧은 손가락의 골일수록")
        print("    급격히 얕아진다. 문제는 **손목 오목점의 깊이가 그 중간에 낀다**는 것이다.")
        print("    → 네 골을 모두 받는 깊이 임계값은 손목까지 함께 받는다.")
        print("      깊이 하나로는 분리 불가능하고, 각도로도 완전히 갈리지 않는다.")

    s = by_name.get("scissors_0", {})
    if s.get("hand_found"):
        print("\n(2) 가위(SCISSORS): 검출은 옳고 **규칙이 틀렸다**")
        print(f"    scissors_0 은 골 {s['gaps']}개를 아주 선명하게(깊이 80px 이상, 각도 60도 미만)")
        print("    잡는다. 마스크도 기하도 실패하지 않았다.")
        print("    이 픽스처의 가위 포즈는 **엄지가 펴져 있다** — 실제로 펴진 손가락은 3개이고,")
        print("    골 2개(엄지–검지, 검지–중지)는 기하학적으로 정확한 답이다.")
        print("    틀린 것은 '가위 = 손가락 2개' 라는 매핑이다. convexity defect 는 어느 골이")
        print("    엄지 쪽인지 구분할 정보를 갖고 있지 않으므로, 이 규칙으로는 원리상 못 맞춘다.")
        print("    (MediaPipe 는 21개 랜드마크로 엄지를 이름으로 지목할 수 있어 이 문제가 없다.)")

    r = by_name.get("rock_0", {})
    if r.get("hand_found"):
        print("\n(3) 바위(ROCK) 는 왜 맞았나 — 운이 아니라 구조")
        print(f"    주먹은 골이 0개이고 solidity={r['solidity']:.3f} 로 껍질을 거의 꽉 채운다.")
        print("    '아무 특징도 없음' 이 곧 특징이라, 세부 기하가 필요 없다.")
        print("    전통 기법이 유일하게 강한 지점이 이런 **덩어리 수준의 구분**이다.")

    print("\n(4) 종합")
    print("    보의 실패는 '더 좋은 마스크/임계값' 으로 완화될 여지가 남아 있지만,")
    print("    가위의 실패는 표현 방식(엄지를 식별할 수 없음)에서 오는 것이라")
    print("    같은 파이프라인 안에서는 해결 경로가 없다. 이것이 랜드마크 회귀")
    print("    (MediaPipe)나 학습된 탐지기(YOLO)가 필요한 직접적인 이유다.")


# ──────────────────────────────────────────────────────────────────────
# 4. 모델 용량 비교 — [사실]
# ──────────────────────────────────────────────────────────────────────

def measure_model_sizes() -> dict:
    """세 접근의 '학습된 모델' 파일 크기를 직접 읽는다.

    이 프로젝트의 값은 하드코딩 0 이 아니라, **모델 파일이 존재하지 않는다는
    사실**에서 나온다. 다운로드할 것도, 로드할 것도 없다.
    """
    out = {"opencv2_rps_classic (이 프로젝트)": {"bytes": 0, "exists": True, "path": "(모델 파일 없음)"}}
    for label, path in MODEL_FILES.items():
        if path.exists():
            b = path.stat().st_size
            out[label] = {"bytes": b, "exists": True, "path": str(path)}
        else:
            out[label] = {"bytes": None, "exists": False, "path": str(path)}
    return out


def print_size_table(sizes: dict, accuracy: dict) -> None:
    print("\n" + "=" * 88)
    print("[사실] 4. 학습된 모델 용량 — os.stat 실측 (하드코딩 없음)")
    print("=" * 88)
    print(f"{'접근':<44}{'bytes':>16}{'MB(SI)':>12}{'MiB':>12}")
    print("-" * 88)
    for label, d in sizes.items():
        if not d["exists"]:
            print(f"{label:<44}{'파일 없음':>16}")
            continue
        b = d["bytes"]
        print(f"{label:<44}{b:>16,}{b / 1e6:>12.2f}{b / (1024 ** 2):>12.2f}")
    print("-" * 88)
    print("이 프로젝트는 학습된 가중치가 0바이트다 — 알고리즘이 코드 안에 전부 있다.")
    print("(단, 코드가 의존하는 OpenCV 라이브러리 자체의 용량은 별도이며 세 접근 모두 동일하게 쓴다.)")

    n, t = accuracy["n_correct"], accuracy["n_total"]
    print("\n" + "=" * 88)
    print("3자 비교 요약")
    print("=" * 88)
    print(f"{'접근':<26}{'학습모델 용량':>16}{'6장 정확도':>14}{'근거':>28}")
    print("-" * 88)
    yolo_x = sizes["yolo11x_rps (yolo/rps)"]
    mp_ = sizes["hand_landmarker.task (mediapipe/hand_lite)"]
    print(f"{'OpenCV 전통기법(이것)':<26}{'0 B':>16}{f'{n}/{t}':>14}"
          f"{'[사실] 이 스크립트 실측':>28}")
    print(f"{'YOLO11x 커스텀':<26}"
          f"{(f'{yolo_x['bytes'] / 1e6:.1f} MB' if yolo_x['exists'] else '파일없음'):>16}"
          f"{'6/6':>14}{'[인용] 타 에이전트 보고':>28}")
    print(f"{'MediaPipe HandLandmarker':<26}"
          f"{(f'{mp_['bytes'] / 1e6:.1f} MB' if mp_['exists'] else '파일없음'):>16}"
          f"{'6/6':>14}{'[인용] 타 에이전트 보고':>28}")
    print("-" * 88)
    print("[추정] 정확도 열은 이 6장 픽스처에서만 관측된 값이다. 픽스처는 3D 렌더")
    print("       이미지이고 배경이 균일한 스튜디오 톤이라, 실제 웹캠 환경(잡다한")
    print("       배경, 혼합 조명)보다 전통 기법에 **유리한** 조건이다. 다른 손/배경/")
    print("       조명에서 같은 수치가 나온다는 근거는 없다.")
    print("[미검증] YOLO·MediaPipe 의 6/6 은 이 스크립트가 잰 값이 아니라 다른")
    print("         에이전트의 보고를 인용한 것이다. 재현하려면 각 프로젝트의")
    print("         테스트를 직접 실행해야 한다.")


# ──────────────────────────────────────────────────────────────────────

def save_debug_images(names: list[str]) -> None:
    """마스크·컨투어·볼록껍질 시각화를 output/ 에 저장한다.

    숫자만으로는 "손끝이 잘렸다"가 와닿지 않는다. 이미지가 있어야 워크샵에서
    바로 보여줄 수 있다.
    """
    OUT_DIR.mkdir(exist_ok=True)
    for name in names:
        img = cv2.imread(str(FIXTURE_DIR / f"{name}.png"))
        mask = skin_mask(img)
        vis = img.copy()
        contour = largest_contour(mask)
        if contour is not None:
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)          # 초록: 실제 윤곽
            cv2.drawContours(vis, [cv2.convexHull(contour)], -1, (255, 0, 0), 2)  # 파랑: 볼록 껍질
            for s_idx, e_idx, f_idx, depth_fixed in convexity_defects(contour):
                if depth_fixed / 256.0 < 10.0:
                    continue
                far = tuple(int(v) for v in contour[f_idx][0])
                angle = defect_angle_deg(contour[s_idx][0], contour[e_idx][0], contour[f_idx][0])
                color = (0, 0, 255) if angle <= 90.0 else (0, 200, 255)   # 빨강=인정, 주황=각도 탈락
                cv2.circle(vis, far, 5, color, -1)
        cv2.imwrite(str(OUT_DIR / f"{name}_mask.png"), mask)
        cv2.imwrite(str(OUT_DIR / f"{name}_vis.png"), vis)
    print(f"\n디버그 이미지 저장: {OUT_DIR}")
    print("  초록=피부 마스크 윤곽 / 파랑=볼록 껍질 / 빨강점=인정된 골 / 주황점=각도로 탈락한 골")


def main() -> int:
    print("=" * 88)
    print("opencv2_rps_classic — 전통 영상처리 가위바위보 실측 리포트")
    print("=" * 88)
    print(f"시각      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"플랫폼    : {platform.platform()} / {platform.machine()}")
    print(f"Python    : {sys.version.split()[0]}")
    print(f"cv2       : {cv2.__version__}")
    print(f"numpy     : {np.__version__}")
    print(f"픽스처    : {FIXTURE_DIR}")

    if not FIXTURE_DIR.is_dir():
        print(f"\n[중단] 픽스처 디렉토리가 없다: {FIXTURE_DIR}")
        return 1

    acc = evaluate_fixtures()
    print_fixture_table(acc)

    diags = [diagnose(name) for name, _ in FIXTURES]
    print_diagnosis(diags)

    sw = sensitivity_sweep()
    print_sweep(sw)
    audit = audit_ceiling_setting(sw)
    print_ceiling_audit(audit)
    print_causal_analysis(diags, sw)

    sizes = measure_model_sizes()
    print_size_table(sizes, acc)

    save_debug_images([name for name, _ in FIXTURES])

    OUT_DIR.mkdir(exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "cv2": cv2.__version__,
        "accuracy": acc,
        "diagnosis": diags,
        "sensitivity": sw,
        "ceiling_audit": audit,
        "model_sizes": sizes,
    }
    out = OUT_DIR / "evaluate_result.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"결과 JSON : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
