"""경량성 벤치마크 — "MediaPipe가 YOLO보다 가볍다"는 주장을 측정으로 판정한다.

이 스크립트는 두 종류의 서로 다른 강도의 주장을 **분리해서** 낸다.

  [사실]  모델 파일 용량. os.path.getsize 로 직접 읽는다. 반박 불가.
  [추정]  추론 속도. 이 기기 1대, 합성/고정 프레임, 특정 조건에서의 관측치.
          다른 기기·다른 입력으로 일반화되지 않는다.

핵심 설계 판단 3가지 (보고서에서 근거를 요구받을 항목):

1. 조건을 3개로 나눈다 (noise / solid / hand).
   팀리드 브리핑은 "노이즈 프레임엔 손이 없어서 랜드마크 회귀를 건너뛰니
   실제보다 빠르게 측정될 것"이라고 편향의 **방향을 단정**했다. 그 단정은
   검증 없이 받아들일 수 없다. MediaPipe VIDEO 모드는 트래킹을 쓰기 때문에
     - 손 없음: 팜 디텍터가 **매 프레임** 돌고, 랜드마크 회귀는 0회
     - 손 있고 트래킹 성공: 팜 디텍터는 거의 안 돌고, 랜드마크 회귀가 1회
   가 된다. 즉 어느 쪽이 비싼지는 사전에 알 수 없다. 편향의 방향조차
   가정이므로, 추측하지 말고 **재서 부호와 크기를 확정**한다.

2. 조건을 라운드로 묶어 라운드 안에서 번갈아 실행한다(randomized block).
   조건을 순차로 다 돌리면 "조건"과 "경과시간"이 교락(confound)된다.
   노트북은 시간이 갈수록 발열로 느려지므로, 마지막에 잰 조건이
   구조적으로 불리해진다. 라운드 = 블록으로 두고 라운드 안에서 조건을
   번갈아 돌리면 발열·백그라운드 부하가 세 조건에 균등하게 실린다.

3. 평균을 대표값으로 쓰지 않는다.
   지연시간은 아래로 막혀 있고(음수 불가) 위로 꼬리가 긴 분포다. GC·스케줄러
   선점 때문에 가끔 튀는 값이 평균을 끌어올린다. 중앙값 + p95 + IQR 로
   보고하고, 불확실성은 **라운드 단위 블록 부트스트랩** CI로 낸다.
   (연속 측정치는 서로 독립이 아니다 — 발열·캐시 상태를 공유한다.
    측정치를 하나씩 iid 재표집하면 CI가 실제보다 좁게 나온다.
    라운드 통째로 재표집해야 라운드 내부 의존성이 보존된다.)

4. 전체 실험을 여러 번 독립 복제한다. ★이 스크립트의 가장 중요한 설계★
   초판은 복제 없이 1회만 돌리고 부트스트랩 CI를 냈다. 그 CI는 noise 조건에
   [29.5, 33.0] ms 로 좁게 나왔다. 그런데 **같은 코드를 그대로 다시 돌리자
   중앙값이 19.0 ms 로 떨어졌다.** CI가 전혀 담지 못한 범위다.
   이유: 부트스트랩이 재표집한 것은 '한 번의 실행 안에서의' 변동뿐이다.
   지배적인 변동원은 실행과 실행 사이에 바뀌는 **주변 CPU 부하**였고,
   한 실행 안에 갇힌 재표집은 그 성분을 원리적으로 볼 수 없다.
   좁은 CI가 정밀함을 뜻하지 않았다 — 분산성분 하나를 통째로 놓친 것이었다.
   그래서 실험 전체를 REPLICATES 회 독립 복제하고,
     (a) 복제 간 중앙값 산포 vs (b) 복제 내 부트스트랩 CI 폭
   을 나란히 보고한다. (a) 가 (b) 보다 크면 CI 를 신뢰구간으로 인용하면 안 된다.

   또한 부하 오염에 강한 통계로 **최솟값**을 함께 낸다. 선점당한 측정은
   실제보다 느려질 뿐 빨라지지 않으므로, 최솟값이 '경합 없는 바닥'에
   가장 가깝다. 중앙값은 부하에 끌려다니지만 최솟값은 훨씬 안정적이다.

실행: ../../../venv/bin/python bench.py   (프로젝트 루트가 아니라 이 디렉토리에서)
"""

from __future__ import annotations

import json
import os
import platform
import random
import resource
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def _find_workshop_root(start: Path) -> Path:
    """yolo/rps가 있는 조상 디렉토리를 찾는다.

    HERE.parent 로 고정하면 이 스크립트를 worktree(.worktrees/docs/bench.py)처럼
    한 단계 더 깊은 곳에서 실행할 때 워크샵 루트를 못 찾는다(.worktrees/를
    가리키게 됨). 실행 위치의 깊이와 무관하게 동작하도록 위로 탐색한다.
    """
    for candidate in (start, *start.parents):
        if (candidate / "yolo" / "rps").is_dir():
            return candidate
    return start.parent


WORKSHOP = _find_workshop_root(HERE)

# 측정 파라미터 — 보고서에 그대로 인용되어야 하는 값들
FRAME_W, FRAME_H = 640, 480
WARMUP_ROUNDS = 15      # 버리는 라운드. 첫 추론은 그래프 초기화/JIT 비용이 섞인다.
MEASURE_ROUNDS = 120    # 복제 1회당 조건별 측정 횟수
REPLICATES = 3          # 독립 복제 실행 횟수. 아래 설계판단 4번 참조.
BOOTSTRAP_RESAMPLES = 2000
SEED = 20260729         # 재현성 고정. 난수 프레임과 부트스트랩 모두 이 시드에 묶인다.

MODEL_PATH = HERE / "models" / "hand_landmarker.task"
YOLO_X = WORKSHOP / "yolo" / "rps" / "models" / "rps_yolo11x_leeyunjai.pt"
YOLO_N = WORKSHOP / "yolo" / "rps" / "models" / "rps_yolo11n_gholamreza.pt"
HAND_FIXTURE = WORKSHOP / "yolo" / "rps" / "tests" / "fixtures" / "rock_0.png"

OUT_DIR = HERE / "output"


# ──────────────────────────────────────────────────────────────────────
# 1. 모델 용량 — [사실]
# ──────────────────────────────────────────────────────────────────────

def measure_model_sizes() -> dict:
    """파일 크기를 직접 읽는다. 하드코딩 없음.

    MB(10^6) 와 MiB(2^20) 를 모두 낸다. 브리핑의 "109MB"는 실제로 MiB 값이고
    SI 기준으로는 114MB다. 배수는 단위와 무관하므로 영향받지 않지만,
    절대값을 인용할 때 단위를 섞으면 5% 오차가 조용히 들어간다.
    """
    targets = {
        "mediapipe_hand_landmarker": MODEL_PATH,
        "yolo11x_rps": YOLO_X,
        "yolo11n_rps": YOLO_N,
    }
    out = {}
    for name, path in targets.items():
        if not path.exists():
            out[name] = {"path": str(path), "exists": False}
            continue
        b = path.stat().st_size
        out[name] = {
            "path": str(path),
            "exists": True,
            "bytes": b,
            "mb_si": b / 1e6,
            "mib": b / (1024 ** 2),
        }
    return out


def print_size_table(sizes: dict) -> None:
    print("\n" + "=" * 78)
    print("[사실] 1. 모델 파일 용량 — os.path.getsize 실측")
    print("=" * 78)
    print(f"{'모델':<32}{'bytes':>14}{'MB(SI)':>11}{'MiB':>11}")
    print("-" * 78)
    for name, d in sizes.items():
        if not d["exists"]:
            print(f"{name:<32}{'파일 없음':>14}")
            continue
        print(f"{name:<32}{d['bytes']:>14,}{d['mb_si']:>11.2f}{d['mib']:>11.2f}")

    mp_ = sizes["mediapipe_hand_landmarker"]
    if not mp_["exists"]:
        return
    base = mp_["bytes"]
    print("-" * 78)
    print("MediaPipe 기준 상대 배수:")
    for name in ("yolo11x_rps", "yolo11n_rps"):
        d = sizes[name]
        if not d["exists"]:
            continue
        r = d["bytes"] / base
        verdict = "MediaPipe가 더 작다" if r > 1 else "MediaPipe가 더 크다  ← 주장 반례"
        print(f"  {name:<26} {r:>7.2f}x   {verdict}")


# ──────────────────────────────────────────────────────────────────────
# 2. 프레임 조건
# ──────────────────────────────────────────────────────────────────────

def build_frames(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """세 조건 모두 640x480 BGR 로 통일한다.

    해상도를 고정해야 '내용'만 처리 조건으로 남는다. 해상도가 같지 않으면
    속도 차이가 내용 때문인지 픽셀 수 때문인지 구분할 수 없다.
    """
    noise = rng.integers(0, 256, size=(FRAME_H, FRAME_W, 3), dtype=np.uint8)
    solid = np.full((FRAME_H, FRAME_W, 3), 128, dtype=np.uint8)

    frames = {"noise": np.ascontiguousarray(noise), "solid": solid}

    img = cv2.imread(str(HAND_FIXTURE))
    if img is not None:
        frames["hand"] = np.ascontiguousarray(
            cv2.resize(img, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)
        )
    return frames


# ──────────────────────────────────────────────────────────────────────
# 3. 엔진 — HandLite 우선, 없으면 MediaPipe 직접
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Engine:
    """조건 1개당 인스턴스 1개.

    조건마다 인스턴스를 분리하는 이유: VIDEO 모드는 프레임 간 트래킹 상태를
    들고 간다. 인스턴스를 공유한 채 조건을 번갈아 넣으면 'hand 프레임에서
    잡은 트래킹'이 다음 noise 프레임 처리에 새어든다. 그러면 각 조건의
    측정치는 그 조건 고유의 비용이 아니라 직전 조건에 오염된 값이 된다.
    """

    label: str
    impl: str                # "HandLite" | "raw-mediapipe"
    _obj: object = None
    _ts: int = 0
    detections: list = field(default_factory=list)

    def detect(self, frame_bgr: np.ndarray) -> int:
        self._ts += 33          # ~30fps 가정. VIDEO 모드는 단조증가 타임스탬프 요구.
        return self._call(frame_bgr, self._ts)

    def _call(self, frame_bgr, ts):
        raise NotImplementedError

    def close(self):
        pass


class HandLiteEngine(Engine):
    def _call(self, frame_bgr, ts):
        return len(self._obj.detect(frame_bgr, ts))

    def close(self):
        self._obj.close()


class RawEngine(Engine):
    def _call(self, frame_bgr, ts):
        import mediapipe as mp
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._obj.detect_for_video(img, ts)
        return len(res.hand_landmarks)

    def close(self):
        self._obj.close()


def make_engine(label: str) -> Engine:
    """HandLite 계약을 먼저 시도하고, 없으면 MediaPipe 직접 경로로 내려간다.

    어느 경로로 쟀는지는 반드시 결과에 라벨로 남긴다. 두 경로는 측정 대상이
    다르다 — HandLite 경로는 래퍼 오버헤드를 포함하고, raw 경로는 엔진만 잰다.
    """
    try:
        from hand_lite.landmarker import HandLite  # noqa
        return HandLiteEngine(label=label, impl="HandLite",
                              _obj=HandLite(model_path=str(MODEL_PATH), num_hands=2))
    except Exception:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        return RawEngine(label=label, impl="raw-mediapipe",
                         _obj=vision.HandLandmarker.create_from_options(opts))


# ──────────────────────────────────────────────────────────────────────
# 4. 통계
# ──────────────────────────────────────────────────────────────────────

def quantile(sorted_vals: list[float], q: float) -> float:
    """선형보간 분위수. numpy 에 맡겨도 되지만 정의를 드러내 둔다."""
    if not sorted_vals:
        return float("nan")
    idx = q * (len(sorted_vals) - 1)
    lo, hi = int(np.floor(idx)), int(np.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def describe(vals: list[float]) -> dict:
    s = sorted(vals)
    return {
        "n": len(s),
        "median_ms": statistics.median(s),
        "mean_ms": statistics.fmean(s),
        "sd_ms": statistics.stdev(s) if len(s) > 1 else 0.0,
        "p25_ms": quantile(s, 0.25),
        "p75_ms": quantile(s, 0.75),
        "p95_ms": quantile(s, 0.95),
        "p99_ms": quantile(s, 0.99),
        "min_ms": s[0],
        "max_ms": s[-1],
    }


def block_bootstrap_median_ci(per_round: list[float], rng: random.Random,
                              resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    """라운드 단위로 재표집한 중앙값의 95% CI.

    측정치를 하나씩 재표집하지 않는 이유: 인접 측정치는 발열/캐시 상태를
    공유하므로 독립이 아니다. iid 재표집은 그 의존성을 무시해 CI를 과하게
    좁게 만든다(= 확신을 부풀린다). 라운드를 통째로 뽑으면 라운드 내부
    상관이 재표집에도 그대로 살아남는다.
    """
    n = len(per_round)
    if n < 2:
        return (float("nan"), float("nan"))
    meds = []
    for _ in range(resamples):
        sample = [per_round[rng.randrange(n)] for _ in range(n)]
        meds.append(statistics.median(sample))
    meds.sort()
    return (quantile(meds, 0.025), quantile(meds, 0.975))


def paired_median_diff_ci(a: list[float], b: list[float], rng: random.Random,
                          resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """같은 라운드에서 잰 a - b 의 짝지은 차이.

    짝을 지어야 하는 이유: 라운드마다 기기 상태(발열, 다른 프로세스)가 다르고
    그건 두 조건에 공통으로 실린다. 짝지으면 그 공통 성분이 상쇄돼 조건
    자체의 효과만 남는다. 짝짓지 않고 두 표본을 따로 비교하면 라운드 간
    변동이 노이즈로 들어가 검정력이 떨어진다.
    """
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    meds = []
    for _ in range(resamples):
        meds.append(statistics.median([d[rng.randrange(n)] for _ in range(n)]))
    meds.sort()
    return {
        "median_diff_ms": statistics.median(d),
        "ci95": [quantile(meds, 0.025), quantile(meds, 0.975)],
        "n_pairs": n,
        "frac_positive": sum(1 for x in d if x > 0) / n,
    }


# ──────────────────────────────────────────────────────────────────────
# 5. 측정 루프
# ──────────────────────────────────────────────────────────────────────

def _one_replicate(frames: dict[str, np.ndarray]) -> tuple[dict, dict, str, float]:
    """블록 설계 1회 실행. 엔진은 매 복제마다 새로 만든다(상태 이월 차단)."""
    conditions = list(frames.keys())
    t_load = time.perf_counter()
    engines = {c: make_engine(c) for c in conditions}
    load_s = time.perf_counter() - t_load
    impl = engines[conditions[0]].impl

    timings: dict[str, list[float]] = {c: [] for c in conditions}
    dets: dict[str, list[int]] = {c: [] for c in conditions}

    for _ in range(WARMUP_ROUNDS):          # 워밍업은 버린다
        for c in conditions:
            engines[c].detect(frames[c])

    for _ in range(MEASURE_ROUNDS):         # 라운드 안에서 조건 번갈아
        for c in conditions:
            t0 = time.perf_counter()
            k = engines[c].detect(frames[c])
            timings[c].append((time.perf_counter() - t0) * 1000.0)
            dets[c].append(k)

    for e in engines.values():
        e.close()
    return timings, dets, impl, load_s


def run_speed(frames: dict[str, np.ndarray]) -> dict:
    conditions = list(frames.keys())
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    reps, det_reps, impl, load_s = [], [], None, 0.0
    loadavg = []
    for r in range(REPLICATES):
        loadavg.append(os.getloadavg())
        t, d, impl, load_s = _one_replicate(frames)
        reps.append(t)
        det_reps.append(d)
        print(f"  복제 {r + 1}/{REPLICATES} 완료  "
              + "  ".join(f"{c}={statistics.median(t[c]):.2f}ms" for c in conditions))
    loadavg.append(os.getloadavg())

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rng = random.Random(SEED)

    stats = {}
    for c in conditions:
        pooled = [x for t in reps for x in t[c]]
        d = describe(pooled)
        d["median_ci95_within_run"] = list(
            block_bootstrap_median_ci(reps[0][c], rng))
        d["replicate_medians"] = [statistics.median(t[c]) for t in reps]
        d["replicate_mins"] = [min(t[c]) for t in reps]
        d["between_replicate_sd"] = (
            statistics.stdev(d["replicate_medians"]) if REPLICATES > 1 else 0.0)
        d["between_replicate_range"] = (
            max(d["replicate_medians"]) - min(d["replicate_medians"]))
        counts = [x for dd in det_reps for x in dd[c]]
        d["hands_detected_frac_nonzero"] = sum(1 for x in counts if x > 0) / len(counts)
        stats[c] = d

    return {
        "impl": impl,
        "load_seconds_per_replicate": load_s,
        "maxrss_delta_mb": (rss_after - rss_before) / 1e6,
        "warmup_rounds": WARMUP_ROUNDS,
        "measure_rounds": MEASURE_ROUNDS,
        "replicates": REPLICATES,
        "frame_size": [FRAME_W, FRAME_H],
        "loadavg_samples": loadavg,
        "per_condition": stats,
        "raw_ms": {c: [x for t in reps for x in t[c]] for c in conditions},
        "raw_by_replicate": {c: [t[c] for t in reps] for c in conditions},
    }


def print_speed(res: dict) -> None:
    print("\n" + "=" * 78)
    print("[추정] 2. 추론 속도 분포 — 이 기기·이 입력·이 순간의 부하 한정")
    print("=" * 78)
    la = res["loadavg_samples"]
    print(f"프레임 {FRAME_W}x{FRAME_H} | 워밍업 {res['warmup_rounds']} 폐기 "
          f"| 조건당 {res['measure_rounds']}회 x {res['replicates']}복제 "
          f"| 엔진 {res['impl']}")
    print(f"CPU 코어 {os.cpu_count()} | load average 1분: "
          + " → ".join(f"{x[0]:.1f}" for x in la))
    print()
    print(f"{'조건':<8}{'n':>6}{'중앙값':>9}{'최솟값':>9}{'p95':>9}{'SD':>8}"
          f"{'최대':>9}{'손검출':>8}")
    print("-" * 78)
    for c, d in res["per_condition"].items():
        print(f"{c:<8}{d['n']:>6}{d['median_ms']:>9.2f}{d['min_ms']:>9.2f}"
              f"{d['p95_ms']:>9.2f}{d['sd_ms']:>8.2f}{d['max_ms']:>9.2f}"
              f"{d['hands_detected_frac_nonzero'] * 100:>7.0f}%")
    print("-" * 78)
    print("단위 ms. n 은 복제 전체를 합친 개수. '손검출' = 손이 1개 이상 잡힌 프레임 비율.")


def print_variance_warning(res: dict) -> None:
    """부트스트랩 CI 가 실제 변동을 담고 있는지 검사한다.

    이 블록이 이 스크립트의 존재 이유다. 좁은 CI 를 그대로 인용하면
    '정밀하게 측정했다'는 인상을 주지만, 복제 간 산포가 CI 폭보다 크면
    그 CI 는 커버리지를 못 내는 숫자다.
    """
    print("\n" + "=" * 78)
    print("[진단] 이 속도 수치를 신뢰구간으로 인용해도 되나 — 분산성분 점검")
    print("=" * 78)
    print(f"{'조건':<8}{'복제별 중앙값':<28}{'복제간 범위':>12}{'복제내 CI폭':>14}{'비율':>8}")
    print("-" * 78)
    worst = 0.0
    for c, d in res["per_condition"].items():
        meds = d["replicate_medians"]
        ci = d["median_ci95_within_run"]
        w_within = ci[1] - ci[0]
        rng_between = d["between_replicate_range"]
        ratio = rng_between / w_within if w_within > 0 else float("inf")
        worst = max(worst, ratio)
        meds_s = ", ".join(f"{m:.1f}" for m in meds)
        print(f"{c:<8}{meds_s:<28}{rng_between:>12.2f}{w_within:>14.2f}{ratio:>8.1f}x")
    print("-" * 78)
    if worst > 1.0:
        print(f"판정: 복제 간 산포가 복제 내 CI 폭의 최대 {worst:.1f}배다.")
        print("  → 부트스트랩 CI 를 신뢰구간으로 인용하면 **안 된다**. 한 번의 실행")
        print("    안에서만 재표집했기 때문에, 실행마다 바뀌는 주변 부하 성분을")
        print("    구조적으로 볼 수 없다. 좁은 CI 는 정밀함이 아니라 시야 좁음이다.")
        print("  → 중앙값 대신 **최솟값**(경합 없는 바닥)을 기기 성능의 대표값으로")
        print("    쓰고, 중앙값은 '이 부하 상태에서의 관측치'로만 인용한다.")
    else:
        print("판정: 복제 간 산포가 CI 폭 이내다. 이 수치는 비교적 안정적이다.")


def print_bias(res: dict) -> None:
    """브리핑이 단정한 편향 방향을 실측으로 판정한다."""
    per = res["per_condition"]
    if "hand" not in per or "noise" not in per:
        print("\n[한계] hand 조건 프레임을 만들지 못해 편향 방향을 측정하지 못했다.")
        return

    rng = random.Random(SEED + 1)
    diff = paired_median_diff_ci(res["raw_ms"]["hand"], res["raw_ms"]["noise"], rng)

    print("\n" + "=" * 78)
    print("[추정] 3. 합성 노이즈 편향 — 방향과 크기를 측정으로 확정")
    print("=" * 78)
    print("짝지은 차이 (hand - noise), 같은 라운드끼리 비교:")
    print(f"  중앙값 차이 : {diff['median_diff_ms']:+.2f} ms")
    print(f"  95% CI      : [{diff['ci95'][0]:+.2f}, {diff['ci95'][1]:+.2f}] ms")
    print(f"  쌍 개수     : {diff['n_pairs']}")
    print(f"  hand가 더 느린 라운드 비율: {diff['frac_positive'] * 100:.1f}%")

    # 최솟값 기준 비교도 함께 낸다. 최솟값은 선점 오염이 가장 적은 통계라
    # 부하가 흔들려도 두 조건의 '바닥 비용' 비율은 비교적 안정적이다.
    mn_h, mn_n = per["hand"]["min_ms"], per["noise"]["min_ms"]
    print(f"\n  경합 없는 바닥(최솟값) 기준: hand {mn_h:.2f} vs noise {mn_n:.2f} ms "
          f"→ {(mn_h / mn_n - 1) * 100:+.1f}%")
    print("  (짝지은 차이는 라운드마다 공통으로 실리는 주변 부하가 상쇄되므로,")
    print("   절대 지연시간보다 부하 오염에 훨씬 강하다. 방향 판정은 이쪽을 믿는다.)")

    lo, hi = diff["ci95"]
    rel = diff["median_diff_ms"] / per["noise"]["median_ms"] * 100
    print()
    if lo > 0:
        print(f"  판정: 노이즈 프레임이 손 있는 프레임보다 **빠르다**. "
              f"노이즈 기준 {rel:+.1f}%.")
        print("  → 브리핑의 우려대로 낙관 편향이다. 노이즈 수치는 실사용보다 빠르다.")
    elif hi < 0:
        print(f"  판정: 노이즈 프레임이 오히려 **느리다**. 노이즈 기준 {rel:+.1f}%.")
        print("  → 브리핑이 단정한 편향 방향이 뒤집혔다. 손이 없으면 팜 디텍터가")
        print("    매 프레임 돌아서, 트래킹이 걸린 손 프레임보다 비쌀 수 있다.")
        print("    즉 노이즈 측정은 낙관이 아니라 **비관** 쪽이다.")
    else:
        print(f"  판정: CI가 0을 포함한다. 이 반복 수로는 방향을 확정할 수 없다.")
        print("  → '노이즈라서 빠르게 나왔다'고 말할 근거도, 반대 근거도 없다.")


# ──────────────────────────────────────────────────────────────────────
# 6. YOLO 속도 — 잴 수 있는지부터 확인
# ──────────────────────────────────────────────────────────────────────

def probe_yolo() -> dict:
    """설치 여부만 확인한다. 없으면 없다고 보고하고 끝낸다.

    새 패키지를 설치하지 않는다(지시). 그리고 설치돼 있었더라도 같은
    노이즈 프레임으로 두 모델을 비교하는 건 공정하지 않다 — 아래 보고 참조.
    """
    out = {}
    for m in ("ultralytics", "torch"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "unknown")
        except Exception as e:
            out[m] = f"MISSING ({type(e).__name__})"
    return out


# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 78)
    print("경량성 벤치마크 — MediaPipe HandLandmarker vs YOLO11 (RPS)")
    print("=" * 78)
    print(f"시각        : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"플랫폼      : {platform.platform()} / {platform.machine()}")
    print(f"Python      : {sys.version.split()[0]}  ({sys.executable})")
    print(f"cv2         : {cv2.__version__}")
    try:
        import mediapipe as mp
        print(f"mediapipe   : {mp.__version__}")
    except Exception as e:
        print(f"mediapipe   : import 실패 {e}")
        return 1
    print(f"난수 시드   : {SEED}")

    sizes = measure_model_sizes()
    print_size_table(sizes)

    if not MODEL_PATH.exists():
        print(f"\n[중단] 모델 파일이 없다: {MODEL_PATH}")
        return 1

    # HandLite 계약 존재 여부를 명시적으로 보고한다.
    try:
        from hand_lite.landmarker import HandLite  # noqa: F401
        print("\nhand_lite.landmarker.HandLite : import 성공 — 계약 경로로 측정한다.")
    except Exception as e:
        print(f"\nhand_lite.landmarker.HandLite : import 실패 ({type(e).__name__}: {e})")
        print("  → 다른 에이전트가 아직 작성 중. MediaPipe 직접 호출로 대체 측정한다.")
        print("    대체 경로는 동일한 .task 모델의 엔진 비용을 재지만,")
        print("    HandLite 래퍼가 추가하는 변환 비용은 포함하지 않는다.")

    rng = np.random.default_rng(SEED)
    frames = build_frames(rng)
    if "hand" not in frames:
        print(f"\n[경고] 손 이미지 fixture 로드 실패: {HAND_FIXTURE}")

    print(f"\n속도 측정 시작 ({REPLICATES}회 독립 복제)...")
    res = run_speed(frames)
    print_speed(res)
    print_variance_warning(res)
    print_bias(res)

    yolo = probe_yolo()
    print("\n" + "=" * 78)
    print("4. YOLO 속도 비교 — 실행 가능성 점검")
    print("=" * 78)
    for k, v in yolo.items():
        print(f"  {k:<14}: {v}")
    if any("MISSING" in str(v) for v in yolo.values()):
        print("\n  → 이 venv 에 ultralytics/torch 가 없다. YOLO 추론 속도는 "
              "**측정하지 않았다**.")
        print("    측정하지 않은 것을 추정으로 메우지 않는다. 속도 우열은 미판정이다.")

    OUT_DIR.mkdir(exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "seed": SEED,
        "sizes": sizes,
        "speed": {k: v for k, v in res.items() if k != "raw_ms"},
        "yolo_probe": yolo,
    }
    out = OUT_DIR / "bench_result.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n결과 JSON: {out}")

    # ── 결론 ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("결론 — 사실과 추정을 섞지 않는다")
    print("=" * 78)
    mpb = sizes["mediapipe_hand_landmarker"]["bytes"]
    if sizes["yolo11x_rps"]["exists"]:
        print(f"[사실] MediaPipe .task 는 yolo11x 대비 "
              f"{sizes['yolo11x_rps']['bytes'] / mpb:.1f}배 작다.")
    if sizes["yolo11n_rps"]["exists"]:
        r = sizes["yolo11n_rps"]["bytes"] / mpb
        print(f"[사실] 그러나 yolo11n 대비로는 {1 / r:.2f}배 **크다**. "
              f"'MediaPipe가 YOLO보다 가볍다'는 무조건 참이 아니다.")
    per = payload["speed"]["per_condition"]
    print(f"[추정] 속도 중앙값은 복제 간에도 흔들린다"
          f"(noise 복제별: {', '.join(f'{m:.1f}' for m in per['noise']['replicate_medians'])} ms). "
          f"단일 수치로 인용 금지.")
    print(f"[추정] 경합 없는 바닥은 noise {per['noise']['min_ms']:.1f} ms"
          + (f", hand {per['hand']['min_ms']:.1f} ms" if "hand" in per else "")
          + " — 기기 성능 대표값으로는 이쪽이 낫다.")
    print("[미판정] YOLO 속도는 측정하지 않았다(ultralytics 부재). 속도 우열은 판정 불가.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
