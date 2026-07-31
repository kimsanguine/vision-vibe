"""웹캠 손인식 진단 도구 — 어디서 막혔는지 단계별로 알려준다.

실행:
    ../../../venv/bin/python doctor.py              # 전체 진단 (카메라 사용)
    ../../../venv/bin/python doctor.py --no-camera  # 1단계만 (카메라 안 켬)

왜 이 파일이 따로 있는가:
    app.py 는 "손이 안 잡힌다"는 상황에서 아무것도 알려주지 않는다. 카메라가
    안 열린 건지, 열렸는데 검은 프레임만 오는 건지, 프레임은 멀쩡한데 모델이
    손을 못 찾는 건지 구분이 안 된다. 이 도구는 그 경계를 하나씩 끊어서
    **어느 단계에서 실패했는지**와 **다음에 무엇을 하면 되는지**를 출력한다.

    5단계(미러링/엄지 진단)는 이 저장소에 남아 있는 미해결 의혹 하나를
    사람 손으로 판정하기 위한 것이다. 자세한 배경은 WEBCAM_VERIFY.md 참조.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "hand_landmarker.task"

# app.py 와 동일한 프레임 간격. VIDEO 모드는 timestamp_ms 가 단조 증가해야 하므로
# 벽시계가 아니라 프레임 인덱스로 만든다(landmarker.py 계약).
FRAME_INTERVAL_MS = 33

# 엄지 판정에 쓰이는 두 랜드마크. gesture.py 의 _THUMB_IP / _THUMB_TIP 과 같은 값이며,
# 진단 출력에 실제 x 좌표를 찍기 위해 여기서도 참조한다.
THUMB_IP = 3
THUMB_TIP = 4

# 프레임이 "균일한 색 한 장"인지 판단하는 표준편차 임계값.
# macOS 카메라 권한 미승인 시 AVFoundation 이 전부 0인 프레임을 돌려주는 사례가 있어
# isOpened()==True 인데도 아무것도 안 보이는 조용한 실패가 생긴다. 그걸 잡는 값.
FLAT_FRAME_STD = 1.0

OK = "[통과]"
NG = "[실패]"
WARN = "[주의]"


# ---------------------------------------------------------------------------
# 출력 헬퍼
# ---------------------------------------------------------------------------

def head(step: str, title: str) -> None:
    print()
    print("=" * 68)
    print(f" {step}. {title}")
    print("=" * 68)


def hint(*lines: str) -> None:
    """실패했을 때 '다음에 무엇을 할지'를 출력한다."""
    print("  ↳ 다음 행동:")
    for line in lines:
        print(f"     - {line}")


# ---------------------------------------------------------------------------
# 프레임 공급원 — 카메라 + 단조 증가 타임스탬프를 한 곳에서 관리
# ---------------------------------------------------------------------------

class FrameSource:
    """cap.read() 와 timestamp_ms 카운터를 묶은다.

    타임스탬프를 여기서 관리하는 이유: landmarker.py 는 timestamp_ms 가
    직전보다 크지 않으면 ValueError 를 던진다(조용한 보정 없음). 단계를
    넘나들며 카운터를 따로 두면 그 계약을 깨기 쉬우므로 한 곳에 모은다.
    """

    def __init__(self, cap, index: int) -> None:
        self.cap = cap
        self.index = index
        self._frame_no = 0

    def next(self):
        """(성공여부, 프레임, timestamp_ms) 를 돌려준다."""
        ok, frame = self.cap.read()
        ts = self._frame_no * FRAME_INTERVAL_MS
        self._frame_no += 1
        return ok, frame, ts

    def release(self) -> None:
        self.cap.release()


# ---------------------------------------------------------------------------
# 1단계 — 환경 점검 (카메라 불필요)
# ---------------------------------------------------------------------------

def step_environment(model_path: Path) -> bool:
    head("1단계", "환경 점검 — 인터프리터 · 패키지 · 모델 파일")

    print(f"  Python      : {platform.python_version()}")
    print(f"  실행 파일   : {sys.executable}")
    print(f"  운영체제    : {platform.system()} {platform.release()} ({platform.machine()})")
    print()

    ok = True

    for name in ("mediapipe", "cv2", "PIL", "numpy"):
        try:
            module = __import__(name)
        except ImportError as exc:
            print(f"  {NG} {name} 임포트 실패 — {exc}")
            hint(
                "워크숍 venv 의 python 으로 실행했는지 확인: ../../../venv/bin/python doctor.py",
                "시스템 python 으로 실행하면 패키지가 없어 여기서 멈춥니다.",
            )
            ok = False
            continue
        version = getattr(module, "__version__", "(버전 정보 없음)")
        print(f"  {OK} {name:<10} {version}")

    print()
    if model_path.is_file():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  {OK} 모델 파일    {model_path} ({size_mb:.1f} MB)")
        if size_mb < 1.0:
            print(f"  {WARN} 크기가 1MB 미만입니다 — 다운로드가 중간에 끊겼을 수 있습니다.")
            hint("모델을 지우고 다시 받으세요 (아래 curl 명령).")
            ok = False
    else:
        print(f"  {NG} 모델 파일 없음: {model_path}")
        hint(
            "프로젝트 루트에서 아래 명령으로 1회 다운로드하세요:",
            'curl -L -o models/hand_landmarker.task '
            '"https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
            'hand_landmarker/float16/1/hand_landmarker.task"',
        )
        ok = False

    return ok


def step_engine_load(model_path: Path) -> bool:
    """2단계 — 카메라 없이 추론 엔진만 띄워본다.

    모델 파일이 '있다'와 '로드된다'는 다른 문제다(손상된 파일, 아키텍처 불일치).
    빈 프레임 1장을 흘려 추론 경로까지 실제로 통과시켜 확인한다. 손은 당연히
    0개가 나오며, 여기서 보는 것은 검출 결과가 아니라 '예외 없이 돌았는가'다.
    """
    head("2단계", "추론 엔진 로드 — 카메라 없이 파이프라인 관통")

    try:
        import numpy as np

        from hand_lite.landmarker import HandLite
    except Exception as exc:  # noqa: BLE001 — 임포트 실패 원인을 그대로 보여준다
        print(f"  {NG} hand_lite 임포트 실패 — {type(exc).__name__}: {exc}")
        hint("프로젝트 루트(app.py 가 있는 디렉터리)에서 실행했는지 확인하세요.")
        return False

    started = time.monotonic()
    try:
        with HandLite(model_path=str(model_path), num_hands=2) as engine:
            load_sec = time.monotonic() - started
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            hands = engine.detect(blank, timestamp_ms=0)
    except Exception as exc:  # noqa: BLE001
        print(f"  {NG} 엔진 로드/추론 실패 — {type(exc).__name__}: {exc}")
        hint(
            "모델 파일이 손상됐을 수 있습니다. 지우고 다시 받아보세요.",
            "그래도 안 되면 이 메시지 전문을 그대로 공유하세요.",
        )
        return False

    print(f"  {OK} 모델 로드 {load_sec:.2f}초")
    print(f"  {OK} 빈 프레임 추론 성공 (검출된 손 {len(hands)}개 — 빈 화면이므로 0이 정상)")
    return True


# ---------------------------------------------------------------------------
# 3단계 — 카메라 열기
# ---------------------------------------------------------------------------

def camera_permission_hint() -> None:
    system = platform.system()
    if system == "Darwin":
        hint(
            "시스템 설정 > 개인정보 보호 및 보안 > 카메라 에서 지금 쓰는 터미널 앱"
            "(터미널 / iTerm / VS Code 등)이 켜져 있는지 확인하세요.",
            "한 번 '거부'를 누른 앱은 팝업이 다시 뜨지 않습니다 — 목록에서 직접 켜야 합니다.",
            "목록에 앱이 아예 없다면 터미널을 완전히 종료했다가 다시 열고 재실행하세요.",
            "FaceTime · Zoom · Photo Booth 등 카메라를 쓰는 앱을 모두 종료하세요.",
        )
    elif system == "Windows":
        hint(
            "설정 > 개인 정보 및 보안 > 카메라 에서 '데스크톱 앱이 카메라에 액세스하도록 허용'을 켜세요.",
            "Zoom · Teams · 카메라 앱 등 카메라를 점유 중인 프로그램을 종료하세요.",
            "노트북이라면 카메라 물리 셔터/키보드 단축키(Fn+카메라)로 꺼져 있지 않은지 확인하세요.",
        )
    else:
        hint(
            "ls /dev/video* 로 장치가 보이는지 확인하세요.",
            "장치가 있는데 안 열리면 사용자가 video 그룹에 속해 있는지 확인하세요"
            "(sudo usermod -aG video $USER 후 재로그인).",
        )


def step_open_camera(indices: list[int]) -> FrameSource | None:
    head("3단계", "카메라 열기")

    import cv2

    for idx in indices:
        print(f"  인덱스 {idx} 시도 중...", flush=True)
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"  {OK} 인덱스 {idx} 열림")
            return FrameSource(cap, idx)
        cap.release()
        print(f"  {NG} 인덱스 {idx} 열리지 않음")

    print()
    print(f"  {NG} 시도한 모든 인덱스({', '.join(map(str, indices))})에서 카메라를 열지 못했습니다.")
    camera_permission_hint()
    return None


# ---------------------------------------------------------------------------
# 4단계 — 프레임 획득 (조용한 실패 감지)
# ---------------------------------------------------------------------------

def step_grab_frames(src: FrameSource, count: int = 30) -> bool:
    """카메라가 '열리기는 했는데 아무것도 안 주는' 상태를 구분해 잡는다."""
    head("4단계", "프레임 획득 — 실제로 그림이 들어오는가")

    import numpy as np

    good = []
    failed = 0
    for _ in range(count):
        ok, frame, _ts = src.next()
        if not ok or frame is None:
            failed += 1
            continue
        good.append(frame)

    print(f"  읽기 시도 {count}회 → 성공 {len(good)}회 / 실패 {failed}회")

    if not good:
        print(f"  {NG} 프레임을 한 장도 읽지 못했습니다(카메라는 열렸는데 데이터가 없음).")
        camera_permission_hint()
        return False

    h, w = good[-1].shape[:2]
    stds = [float(f.std()) for f in good]
    means = [float(f.mean()) for f in good]
    motion = float(np.mean(np.abs(good[-1].astype(np.int16) - good[0].astype(np.int16))))

    print(f"  해상도       : {w} x {h}")
    print(f"  밝기 평균    : {min(means):.1f} ~ {max(means):.1f} (0=완전 검정, 255=완전 흰색)")
    print(f"  픽셀 표준편차: {min(stds):.2f} ~ {max(stds):.2f}")
    print(f"  첫↔끝 프레임 평균 변화량: {motion:.2f}")

    if max(stds) < FLAT_FRAME_STD:
        print()
        print(f"  {NG} 모든 프레임이 단색입니다 — 카메라는 열렸지만 영상이 없습니다.")
        print("     macOS에서 카메라 권한이 없을 때 나타나는 대표적인 조용한 실패입니다")
        print("     (열기는 성공하고 검은 프레임만 계속 돌려줌).")
        camera_permission_hint()
        hint("렌즈 커버가 닫혀 있거나 방이 완전히 어두운 경우에도 같은 값이 나옵니다 — 함께 확인하세요.")
        return False

    if motion < 0.5:
        print()
        print(f"  {WARN} 프레임이 변하지 않습니다(정지 화면일 수 있음). 손을 흔들며 다시 실행해 보세요.")

    if failed:
        print(f"  {WARN} 읽기 실패가 {failed}회 있었습니다 — 간헐적이면 USB 연결/대역폭을 의심하세요.")

    print()
    print(f"  {OK} 정상적인 영상 프레임이 들어옵니다.")
    return True


# ---------------------------------------------------------------------------
# 5단계 — 손 검출
# ---------------------------------------------------------------------------

def step_detect_hand(src: FrameSource, engine, max_frames: int, show) -> bool:
    head("5단계", "손 검출 — 카메라에 손을 보여주세요")

    import cv2

    print(f"  손바닥을 카메라에 보여주세요. 최대 {max_frames}프레임(약 {max_frames / 30:.0f}초) 기다립니다.")
    print("  (창이 떠 있으면 [Q] 로 중단할 수 있습니다)")
    print()

    for i in range(max_frames):
        ok, frame, ts = src.next()
        if not ok or frame is None:
            continue
        frame = cv2.flip(frame, 1)  # app.py:116 과 동일한 거울 모드
        hands = engine.detect(frame, ts)

        if show(frame, "doctor - 5단계 손 검출"):
            print(f"  {WARN} 사용자가 중단했습니다.")
            return False

        if hands:
            print(f"  {OK} {i + 1}번째 프레임에서 손 {len(hands)}개 검출")
            for n, hand in enumerate(hands, 1):
                print(f"       손 {n}: handedness={hand.handedness}, score={hand.score:.3f}")
            return True

    print(f"  {NG} {max_frames}프레임 동안 손을 한 번도 검출하지 못했습니다.")
    hint(
        "손 전체가 화면 안에 들어오게 하고 카메라에서 30~60cm 거리를 두세요.",
        "조명을 밝게 하세요 — 역광(창문 앞)이면 실루엣만 잡혀 검출이 어렵습니다.",
        "--frames 300 으로 대기 시간을 늘려 다시 시도해 보세요.",
        "4단계는 통과했으므로 카메라 문제는 아닙니다. 조명·거리·손 자세를 먼저 바꿔보세요.",
    )
    return False


# ---------------------------------------------------------------------------
# 6단계 — 미러링 / 엄지 부호 진단
# ---------------------------------------------------------------------------

class PoseSample:
    """한 자세에 대해 여러 프레임을 모아 다수결로 정리한 관측치."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.handedness: list[str] = []
        self.thumb_up: list[bool] = []
        self.gestures: list[str] = []
        self.dx: list[float] = []  # thumb_tip.x - thumb_ip.x

    def add(self, hand, thumb_up: bool, gesture: str) -> None:
        self.handedness.append(hand.handedness)
        self.thumb_up.append(thumb_up)
        self.gestures.append(gesture)
        self.dx.append(hand.landmarks[THUMB_TIP].x - hand.landmarks[THUMB_IP].x)

    @property
    def n(self) -> int:
        return len(self.handedness)

    def majority_handedness(self) -> str:
        return Counter(self.handedness).most_common(1)[0][0] if self.handedness else "-"

    def majority_gesture(self) -> str:
        return Counter(self.gestures).most_common(1)[0][0] if self.gestures else "-"

    def thumb_up_ratio(self) -> float:
        return sum(self.thumb_up) / self.n if self.n else 0.0

    def mean_dx(self) -> float:
        return sum(self.dx) / self.n if self.n else 0.0

    def line(self) -> str:
        if not self.n:
            return f"    {self.label:<22} 손 검출 안 됨"
        return (
            f"    {self.label:<22} handedness={self.majority_handedness():<5} "
            f"엄지폄={self.thumb_up_ratio():>4.0%}  "
            f"제스처={self.majority_gesture():<5} "
            f"(tip.x-ip.x={self.mean_dx():+.3f}, {self.n}프레임)"
        )


def collect_pose(src: FrameSource, engine_flip, engine_raw, hold: int, show):
    """한 자세를 hold 프레임만큼 관측해 (거울프레임 결과, 원본프레임 결과) 를 돌려준다.

    같은 순간의 프레임을 뒤집은 것과 안 뒤집은 것 두 갈래로 각각 추론한다.
    app.py 는 뒤집은 쪽만 쓰지만, 두 결과를 나란히 봐야 '뒤집기가 판정을
    바꾸는가'를 사람이 직접 확인할 수 있다.
    """
    import cv2

    from hand_lite.gesture import classify, fingers_up

    flipped = PoseSample("거울(app.py 경로)")
    raw = PoseSample("원본(뒤집지 않음)")

    for _ in range(hold):
        ok, frame, ts = src.next()
        if not ok or frame is None:
            continue

        mirror = cv2.flip(frame, 1)
        for engine, sample, img in ((engine_flip, flipped, mirror), (engine_raw, raw, frame)):
            for hand in engine.detect(img, ts):
                thumb = fingers_up(hand)[0]
                sample.add(hand, thumb, classify(hand).value)

        if show(mirror, "doctor - 6단계 미러링 진단"):
            break

    return flipped, raw


def countdown(src: FrameSource, seconds: int, show) -> None:
    """카메라를 계속 읽으면서 카운트다운한다.

    time.sleep 을 쓰지 않는 이유: 그동안 카메라 버퍼에 오래된 프레임이 쌓여
    자세를 바꾸기 전의 장면을 측정하게 된다. 계속 읽어서 버퍼를 비운다.
    """
    import cv2

    for remain in range(seconds, 0, -1):
        print(f"    {remain}...", end="", flush=True)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            ok, frame, _ts = src.next()
            if ok and frame is not None:
                show(cv2.flip(frame, 1), "doctor - 6단계 미러링 진단")
    print(" 측정!")


def step_mirror_diagnosis(src: FrameSource, engine_flip, engine_raw, hold: int, show) -> bool:
    head("6단계", "미러링 / 엄지 부호 진단")

    print("  배경: app.py:116 은 cv2.flip 으로 좌우를 뒤집은 프레임을 판정에 넘깁니다.")
    print("        그런데 gesture.py 는 '입력이 뒤집히지 않았다고 가정'한다고 적혀 있고,")
    print("        그 보정 코드는 app.py 어디에도 없습니다. 엄지 부호가 뒤집혔는지")
    print("        실제 손으로만 확인할 수 있습니다.")
    print()
    print("  핵심: 엄지 부호가 뒤집혔다면 **주먹이 '바위'가 아니라 '판정불가'로 나옵니다.**")
    print("        (classify 의 바위 조건은 엄지를 포함한 5개 전부 '굽음'이기 때문)")
    print()
    print("  요령: 손바닥이 카메라를 향하게 하세요(손등이 아니라). 가위바위보 낼 때 자세입니다.")
    print()

    poses = [
        ("오른손", "손바닥을 활짝 펴서(보) 손바닥이 카메라를 향하게", "Right", "보"),
        ("오른손", "주먹을 쥐어서(바위) 엄지를 손가락 위에 얹어", "Right", "바위"),
        ("왼손", "손바닥을 활짝 펴서(보) 손바닥이 카메라를 향하게", "Left", "보"),
    ]

    results = []
    for hand_name, how, expect_handedness, expect_gesture in poses:
        print("-" * 68)
        print(f"  ▶ {hand_name}을 {how} 들어주세요.")
        try:
            input("    준비되면 Enter를 누르세요 (건너뛰려면 Ctrl+C) > ")
        except (EOFError, KeyboardInterrupt):
            print("\n  진단을 중단했습니다.")
            return False

        countdown(src, 3, show)
        flipped, raw = collect_pose(src, engine_flip, engine_raw, hold, show)
        print(flipped.line())
        print(raw.line())
        print()
        results.append((hand_name, expect_handedness, expect_gesture, flipped, raw))

    return verdict(results)


def verdict(results) -> bool:
    """관측치로부터 판정을 내린다. 관측 못 한 것은 판정하지 않는다."""
    print("=" * 68)
    print("  판정")
    print("=" * 68)

    measured = [r for r in results if r[3].n > 0]
    if not measured:
        print(f"  {NG} 어떤 자세에서도 손이 검출되지 않아 판정할 수 없습니다.")
        hint("조명을 밝게 하고 손을 화면 중앙에 크게 잡히도록 한 뒤 다시 실행하세요.")
        return False

    if len(measured) < len(results):
        skipped = [r[0] + "/" + r[2] for r in results if r[3].n == 0]
        print(f"  {WARN} 손이 안 잡힌 자세가 있어 그 항목은 판정에서 제외합니다: {', '.join(skipped)}")
        print()

    problems = []

    # (a) handedness 라벨이 실제로 든 손과 일치하는가
    for hand_name, expect_handedness, expect_gesture, flipped, _raw in measured:
        got = flipped.majority_handedness()
        mark = OK if got == expect_handedness else NG
        print(f"  {mark} {hand_name} {expect_gesture} → 거울 프레임의 handedness = '{got}' "
              f"(기대값 '{expect_handedness}')")
        if got != expect_handedness:
            problems.append(
                f"{hand_name} {expect_gesture} 에서 handedness 라벨이 실제 손과 반대입니다"
                f"('{got}'). MediaPipe 가 뒤집힌 입력을 전제로 라벨을 붙이는지가 여기서 갈립니다."
            )

    print()

    # (b) 제스처가 실제로 낸 모양과 일치하는가 — 엄지 부호의 결정적 증거
    for hand_name, _eh, expect_gesture, flipped, raw in measured:
        got = flipped.majority_gesture()
        mark = OK if got == expect_gesture else NG
        print(f"  {mark} {hand_name} {expect_gesture} → 판정 '{got}' "
              f"(엄지 폄 비율 {flipped.thumb_up_ratio():.0%})")
        if got != expect_gesture:
            problems.append(f"{hand_name} {expect_gesture} 가 '{got}' 로 판정됩니다.")

    # (c) 주먹인데 엄지가 '폄'으로 나오면 부호 반전 확정
    fist = next((r for r in measured if r[2] == "바위"), None)
    print()
    if fist and fist[3].thumb_up_ratio() > 0.5:
        print(f"  {NG} 주먹을 쥐었는데 엄지가 '폄'으로 판정됩니다"
              f"({fist[3].thumb_up_ratio():.0%} 프레임).")
        print("     → 엄지 판정 부호가 뒤집혔다는 결정적 증거입니다.")
    elif fist:
        print(f"  {OK} 주먹을 쥐었을 때 엄지가 '굽음'으로 판정됩니다 — 엄지 부호는 정상입니다.")

    # (d) 뒤집기가 판정을 바꾸는가
    print()
    for hand_name, _eh, expect_gesture, flipped, raw in measured:
        if raw.n == 0:
            continue
        same = flipped.majority_gesture() == raw.majority_gesture()
        tag = "동일" if same else "다름"
        print(f"  [참고] {hand_name} {expect_gesture}: 거울='{flipped.majority_gesture()}' "
              f"vs 원본='{raw.majority_gesture()}' → {tag}")

    print()
    print("-" * 68)
    if problems:
        print(f"  {NG} 판정: 문제 있음")
        for p in problems:
            print(f"     · {p}")
        hint(
            "위 출력 전체를 그대로 복사해 공유하세요 — 이 수치가 수정 방향을 결정합니다.",
            "WEBCAM_VERIFY.md 의 '결과별 다음 행동' 절을 보세요.",
        )
        return False

    print(f"  {OK} 판정: 관측한 자세 범위에서 app.py 의 거울 모드 + gesture.py 조합은 정상 동작합니다.")
    print("     (손바닥이 카메라를 향한 자세에 한정된 결과입니다. 손등을 보이거나")
    print("      손을 옆으로 눕히면 gesture.py 가 명시한 한계로 오판정될 수 있습니다.)")
    return True


# ---------------------------------------------------------------------------
# 엔트리포인트
# ---------------------------------------------------------------------------

def make_show(enabled: bool):
    """미리보기 창 출력 함수를 만든다. 반환값 True 는 '사용자가 Q로 중단'.

    GUI 가 없는 환경(헤드리스 서버, 일부 원격 세션)에서는 cv2.imshow 가 예외를
    던진다. 첫 호출에서 한 번 걸리면 이후로는 창 없이 진행한다 — 창이 없다고
    진단 자체를 못 할 이유는 없기 때문이다.
    """
    if not enabled:
        return lambda frame, title: False

    import cv2

    state = {"alive": True}

    def show(frame, title: str) -> bool:
        if not state["alive"]:
            return False
        try:
            cv2.imshow(title, frame)
            return (cv2.waitKey(1) & 0xFF) == ord("q")
        except cv2.error as exc:
            print(f"  {WARN} 미리보기 창을 열 수 없어 창 없이 계속합니다 ({exc.__class__.__name__}).")
            state["alive"] = False
            return False

    return show


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="웹캠 손인식 진단 도구 — 어느 단계에서 막혔는지 알려준다",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  ../../../venv/bin/python doctor.py                # 전체 진단\n"
            "  ../../../venv/bin/python doctor.py --no-camera    # 카메라 안 켜고 환경만 점검\n"
            "  ../../../venv/bin/python doctor.py --camera-index 1\n"
            "  ../../../venv/bin/python doctor.py --frames 300   # 손 검출을 더 오래 기다림\n"
        ),
    )
    p.add_argument("--no-camera", action="store_true",
                   help="카메라를 열지 않고 1~2단계(환경·엔진)만 실행한다")
    p.add_argument("--camera-index", type=int, default=None,
                   help="특정 카메라 인덱스만 시도한다. 기본값: 0, 1, 2 순차 시도")
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="HandLandmarker .task 모델 경로")
    p.add_argument("--frames", type=int, default=150,
                   help="5단계에서 손을 기다릴 최대 프레임 수. 기본값: 150(약 5초)")
    p.add_argument("--hold", type=int, default=30,
                   help="6단계에서 자세 하나당 관측할 프레임 수. 기본값: 30")
    p.add_argument("--no-window", action="store_true",
                   help="미리보기 창을 띄우지 않는다(헤드리스 환경)")
    p.add_argument("--skip-mirror", action="store_true",
                   help="6단계(미러링 진단)를 건너뛴다")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)

    print()
    print("  경량 MediaPipe 손인식 — 웹캠 진단 도구")
    print("  각 단계는 통과/실패가 명확하며, 실패하면 다음 행동을 알려줍니다.")

    if not step_environment(model_path):
        print(f"\n  {NG} 1단계에서 중단합니다. 위 안내를 먼저 해결하세요.")
        return 1

    if not step_engine_load(model_path):
        print(f"\n  {NG} 2단계에서 중단합니다.")
        return 1

    if args.no_camera:
        print()
        print("=" * 68)
        print(f"  {OK} --no-camera 모드: 카메라를 켜지 않는 단계는 모두 통과했습니다.")
        print("     카메라 · 손 검출 · 미러링 진단은 아직 확인되지 않았습니다.")
        print("     --no-camera 없이 다시 실행해 3~6단계를 마저 진행하세요.")
        print("=" * 68)
        return 0

    indices = [args.camera_index] if args.camera_index is not None else [0, 1, 2]

    import cv2

    from hand_lite.landmarker import HandLite

    src = step_open_camera(indices)
    if src is None:
        return 1

    engine_flip = None
    engine_raw = None
    show = make_show(not args.no_window)
    exit_code = 0

    try:
        if not step_grab_frames(src):
            return 1

        # 거울 프레임용과 원본 프레임용 엔진을 따로 둔다. VIDEO 모드는 직전 프레임의
        # 추적 상태를 재사용하므로, 한 엔진에 뒤집힌/안 뒤집힌 프레임을 번갈아 넣으면
        # 추적이 매 프레임 깨져 두 경로 모두 나쁜 결과가 나온다.
        engine_flip = HandLite(model_path=str(model_path), num_hands=1)
        engine_raw = HandLite(model_path=str(model_path), num_hands=1)

        if not step_detect_hand(src, engine_flip, args.frames, show):
            return 1

        if args.skip_mirror:
            print()
            print(f"  {WARN} --skip-mirror 로 6단계를 건너뛰었습니다. 미러링 의혹은 미확인 상태입니다.")
        elif not step_mirror_diagnosis(src, engine_flip, engine_raw, args.hold, show):
            exit_code = 1

    except KeyboardInterrupt:
        print("\n\n  사용자가 중단했습니다(Ctrl+C).")
        exit_code = 130
    finally:
        # 카메라 해제는 어떤 경로로 끝나든 반드시 실행한다. 놓치면 다음 실행에서
        # '다른 앱이 카메라를 점유 중' 처럼 보이는 유령 증상이 생긴다.
        src.release()
        for engine in (engine_flip, engine_raw):
            if engine is not None:
                engine.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # macOS 에서 창이 실제로 닫히려면 이벤트 루프가 한 번 더 돌아야 한다
        print("\n  [정리] 카메라와 모델 리소스를 해제했습니다.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
