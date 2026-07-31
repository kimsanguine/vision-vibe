# 경량 MediaPipe 손인식 — Hand Lite

웹캠(또는 정지 이미지) 속 손을 MediaPipe HandLandmarker로 추적하고, 21개 랜드마크
좌표만으로 가위바위보 손모양을 판정하는 최소 구성 데모입니다.
"OpenCV/YOLO 비전 앱, 바이브 코딩으로 만들기" 워크샵의 Part 3 확장 자료이며,
같은 워크샵의 `yolo/rps`(YOLO 방식 가위바위보 인식)와 짝을 이룹니다 — "학습 없이
구글 사전학습 모델을 그대로 붙이는 감각"과 "내 문제 전용으로 모델을 새로 구해와
학습시키는 감각"을 나란히 비교하기 위한 대조군입니다.

```
프레임 → HandLandmarker(엔진) → 21점 좌표 → 기하학적 if문(판정) → 화면 표시
```

---

## 빠른 시작

```bash
cd mediapipe/hand_lite

python3 -m venv venv
venv/bin/pip install -r requirements.txt

# 모델 다운로드 불필요 — models/hand_landmarker.task(7.5MB)가 저장소에 이미 포함되어 있음
venv/bin/python app.py --image <이미지경로>     # 정지 이미지 모드(카메라 불필요)
venv/bin/python app.py                          # 웹캠 모드(손 랜드마크 + 제스처 라벨만)

venv/bin/python rps_game.py --image <이미지경로> # 가위바위보 게임을 이미지 1장으로 실행
venv/bin/python rps_game.py                      # 가위바위보 게임 웹캠 모드(AI와 대결, 점수판)

venv/bin/python doctor.py --no-camera            # 웹캠 없이 환경·엔진만 진단
venv/bin/python doctor.py                        # 카메라 열기부터 미러링 진단까지 6단계 전체 진단
```

`app.py`는 손 추적만 보여주는 최소 데모이고, `rps_game.py`는 그 위에 실제
가위바위보 승부(점수판·제스처 안정화)를 얹은 것입니다. `doctor.py`는 웹캠
경로가 막혔을 때 어느 단계에서 실패했는지 짚어주는 진단 도구입니다 —
자세한 사용법은 [`WEBCAM_VERIFY.md`](WEBCAM_VERIFY.md) 참고.

같은 워크샵의 `yolo/rps`는 114MB 가중치를 별도 스크립트(`fetch_assets.py`)로
내려받아야 실행할 수 있습니다. 이 프로젝트는 모델이 7.5MB로 충분히 작아 **clone
직후 추가 다운로드 없이 바로 실행**됩니다 — 아래 [벤치마크](#벤치마크--모델-용량은-사실-속도는-추정) 절에서
이 크기 차이를 실측으로 확인합니다.

### 실행 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--image` | 없음 | 지정하면 웹캠 대신 이미지 1장을 처리해 `--out` 디렉터리에 저장 |
| `--out` | `output` | 정지 이미지 모드 결과 저장 디렉터리 |
| `--model` | `models/hand_landmarker.task` | HandLandmarker 모델 경로 |
| `--num-hands` | `2` | 동시 추적할 최대 손 개수 |

웹캠 모드 조작: **Q** 종료.

---

## 아키텍처 — 계약으로 나눈 4개 모듈

```
types.py        모듈 간 공유 계약 (Landmark, HandResult, Gesture)
   ↑
landmarker.py   엔진: MediaPipe만 안다 (cv2로 색공간 변환은 하지만 그리기는 안 함)
gesture.py      판정: 아무것도 import하지 않는다 (MediaPipe도 OpenCV도 카메라도)
hud.py          렌더: OpenCV+PIL만 안다 (MediaPipe는 모른다)
app.py          진입점: 위 셋을 엮는다 (--image / 웹캠 두 모드)
bench.py        모델 용량·추론 속도 측정
docs/           prd.md · progress.md · architecture.md
```

`types.py`가 계약입니다. `landmarker.py`/`gesture.py`/`hud.py`는 서로의 내부 구현을
몰라도 되고, 오직 `HandResult`/`Gesture` 같은 계약 타입만 주고받습니다.

**핵심은 `gesture.py`가 순수 함수라는 것**입니다. 입력은 `HandResult`(21개 좌표) 하나,
출력은 `Gesture` 하나뿐이고, MediaPipe·OpenCV·카메라 무엇도 import하지 않습니다.
"손가락이 펴졌는가"는 손끝 좌표와 두 번째 관절 좌표를 비교하면 답이 나오는
문제이므로, 여기에 학습된 분류기를 넣을 이유가 없습니다 — if문으로 충분한 곳에
모델을 쓰지 않는다는 설계 판단이 코드 자체에 주석으로 남아 있습니다. 그 덕에
`gesture.py`는 카메라도 실제 손도 없이 **합성 좌표만으로 100% 테스트**됩니다
(`tests/test_gesture.py` — [테스트](#테스트) 절 참고).

반대로 `landmarker.py`(MediaPipe 실추론)와 `hud.py`(화면 렌더)는 근본적으로
카메라나 모델 파일 없이는 검증할 수 없는 부분이 남습니다. 이 프로젝트는 그
경계를 감추지 않고 [알려진 한계](#알려진-한계정직하게) 절에 명시합니다.

### 더 깊은 문서

| 문서 | 내용 |
|---|---|
| [`docs/prd.md`](docs/prd.md) | 왜 만들었나 — 문제·범위·성공지표·DoD·잔여 리스크 |
| [`docs/architecture.md`](docs/architecture.md) | 어떻게 구성했나 — 계약 경계·데이터 흐름·좌표계 규약·설계 결정의 근거 |
| [`docs/progress.md`](docs/progress.md) | 무슨 일이 있었나 — 타임라인·발견된 결함 3건·실측 수치·남은 일 |

---

## YOLO 방식과의 대비

| 항목 | 이 프로젝트 (MediaPipe) | `yolo/rps` (YOLO) |
|---|---|---|
| 학습 데이터 | 불필요 — 구글이 배포한 사전학습 모델을 그대로 사용 | 필요 — Roboflow *Rock Paper Scissors SXSW* 데이터셋으로 학습된 가중치 |
| 파인튜닝 | 불필요(감지 대상이 손이라는 범용 태스크) | 필요(가위바위보 3클래스는 COCO 기본 모델에 없음) |
| 즉시 사용 가능성 | 모델이 저장소에 포함, clone 직후 실행 | 114MB 가중치를 `fetch_assets.py`로 별도 다운로드해야 실행 |
| 판정 방식 | 21점 좌표 기하학 if문(`gesture.py`, 순수 함수) | 학습된 분류기가 클래스를 직접 출력 |
| 검출 신뢰도 표시 | handedness(왼손/오른손) 분류 확률 — Tasks API는 별도의 손 검출 점수를 주지 않음 | 클래스별 confidence score |

---

## 벤치마크 — 모델 용량은 [사실], 속도는 [추정]

`bench.py`를 이 저장소 안에서 직접 실행해 얻은 수치입니다(2026-07-29, Apple M3 Pro,
mediapipe 1.0.0). 재현: `../../../venv/bin/python bench.py` (`mediapipe/hand_lite/`
디렉토리에서 실행 — 이유는 [트러블슈팅](#트러블슈팅)의 `WORKSHOP` 경로 항목 참고).

### 모델 파일 용량 — `os.path.getsize` 실측, [사실]

| 모델 | 용량 |
|---|---|
| `hand_landmarker.task` (MediaPipe, 이 프로젝트) | **7.82 MB** (7,819,105 bytes) |
| `rps_yolo11x_leeyunjai.pt` (YOLO, `yolo/rps` 채택 모델) | **114.42 MB** (114,420,626 bytes) |
| `rps_yolo11n_gholamreza.pt` (YOLO, `yolo/rps`가 실측 후 탈락시킨 모델) | **5.47 MB** (5,465,619 bytes) |

**"MediaPipe가 YOLO보다 가볍다"는 절반만 참입니다.** 채택된 YOLO 모델(11x)과
비교하면 MediaPipe가 **14.63배 작지만**, 더 작은 YOLO 모델(11n)과 비교하면
오히려 MediaPipe가 **약 1.43배 더 큽니다.** 비교 대상을 무엇으로 잡느냐에 따라
결론이 뒤집힙니다 — 벤치마크 수치를 볼 때는 항상 "무엇과 비교했는가"를 먼저
확인해야 한다는 것 자체가 이 실측에서 얻는 가장 값진 교훈입니다.

여기에 한 겹 더 있습니다. `yolo/rps/README.md`에 따르면 그 5.47MB짜리 11n
모델은 **3 epoch만 학습된 토이 체크포인트**라 정지 이미지·영상 프레임에서
**실제로는 0건 검출**되어 `yolo/rps` 자신이 채택을 거부한 모델입니다. 즉 이
"MediaPipe보다 작은 YOLO 모델"은 크기만 작을 뿐 실사용 가능한 비교 대상이
아닙니다 — 크기 비교에서도 "그 모델이 실제로 동작하는가"를 먼저 물어야
함정에 빠지지 않습니다.

### 추론 속도 — 이 기기·이 순간의 부하 한정, [추정]

| 조건 | n | 중앙값 | 최솟값 | p95 | 손 검출률 |
|---|---|---|---|---|---|
| noise(무작위 노이즈 프레임) | 360 | 43.27 ms | 17.40 ms | 144.37 ms | 0% |
| solid(단색 프레임) | 360 | 43.22 ms | 20.88 ms | 133.92 ms | 0% |

**위 표를 측정한 실행에서는 "손" 조건(실제 손이 찍힌 사진)이 빠져 있습니다.**
당시 `bench.py`는 워크샵 루트를 `HERE.parent`로 고정 계산해서, `.worktrees/docs/`
처럼 한 단계 깊은 위치에서 실행하면 `yolo/rps/tests/fixtures/rock_0.png`를
찾지 못했습니다(콘솔에 `can't open/read file` 경고).

**이 경로 문제는 이후 수정됐습니다**(커밋 `6e02bbd` — 조상 디렉토리를 위로
탐색하는 `_find_workshop_root`). 지금은 worktree 안에서 실행해도 손 조건까지
측정됩니다. 아래 "손 조건 포함" 측정치를 참고하세요.

측정하지 못한 것을 추정으로 메우지 않고 미측정으로 남긴 판단 자체는 그대로
유효합니다 — 값이 없을 때 그럴듯한 숫자를 지어내는 것이 가장 나쁜 선택입니다.

측정 당시 이 기기의 1분 load average는 **651.9 → 991.1 → 802.8 → 661.9**로
극단적으로 높았습니다(12코어 기기 기준 정상 범위를 크게 벗어남 — 다른 무거운
프로세스가 동시에 돌고 있었을 가능성이 높습니다). `bench.py`는 이런 상황을
스스로 진단합니다: 3회 독립 복제의 중앙값이 noise 조건에서 53.3 / 40.7 /
37.1 ms로 흩어졌고, 이 복제 간 산포(16.2ms)가 한 번의 실행 안에서 낸
부트스트랩 신뢰구간 폭(8.4ms)보다 1.9배 넓었습니다. `bench.py`의 판정 그대로
인용하면: **"부트스트랩 CI를 신뢰구간으로 인용하면 안 된다. 좁은 CI는 정밀함이
아니라 시야 좁음이다."** 이 표의 중앙값은 "이 부하 상태에서의 관측치" 그
이상도 이하도 아니며, 기기 성능의 대표값에 조금 더 가까운 쪽은 경합이 가장
적었던 최솟값(noise 17.40ms)입니다.

같은 이유로 YOLO 추론 속도는 이번 실행에서 **측정되지 않았습니다** — 이
`venv`에 `ultralytics`/`torch`가 설치돼 있지 않기 때문입니다(`yolo/rps`는
별도 venv를 씁니다). 측정하지 않은 것을 추정으로 메우지 않습니다. **MediaPipe와
YOLO의 속도 우열은 이 문서 기준으로 미판정입니다.**

### 제스처 정확도 — 실사진 6장 기준 6/6, [사실]

`../../yolo/rps/tests/fixtures/`의 가위바위보 실사진 6장에서 **6/6**입니다.
`tests/test_gesture.py::TestRealPhotoRegression`에 회귀 테스트로 고정돼 있습니다.

이 수치는 처음부터 나온 게 아닙니다. **합성 테스트 42개를 전부 통과한 상태에서
실사진 정확도는 2/6이었습니다.** 원인은 엄지를 `handedness` + x부호로 판정한 것
(엄지 단독 정확도 0/6). 거리비 방식으로 바꿔 6/6이 됐습니다. 아래 "겪은 함정" 참고.

### VIDEO 모드가 실제로 벌어주는 것 — 조건부

`landmarker.py`는 `running_mode=VIDEO`를 씁니다. 그 이득은 **손이 화면에 있을
때만** 나타납니다. `rps_video.avi` 150프레임 × 3복제 측정:

| 조건 | IMAGE | VIDEO | 차이 |
|---|---:|---:|---|
| **손 검출률** (실제 손 동영상) | 50.7% | **68.0%** | **+17.3%p** — [사실] |
| 중앙값 지연 | 235.96 ms | 80.01 ms | −66% — [추정] |
| 최솟값(경합 없는 바닥) | 38.39 ms | 25.42 ms | −34% — [추정] |
| 손 **없는** 노이즈 프레임 | 29.93 ms | 32.01 ms | **VIDEO가 7% 느림** |

**검출률 +17%p가 가장 단단한 근거입니다** — 시간과 달리 시스템 부하에 흔들리지
않습니다. VIDEO 모드는 이전 프레임의 손을 되먹여(`PreviousLoopbackCalculator`)
팜 디텍터가 놓친 프레임을 이어붙입니다. 되먹일 손이 없으면 그 경로가 아예 돌지
않고 타임스탬프 동기화 비용만 남아, **손이 없을 때는 오히려 손해**입니다.

즉 "VIDEO 모드가 더 가볍다"는 무조건 참이 아니라 **손이 있을 때 참**입니다.

---

## 알려진 한계(정직하게)

- **실제 웹캠으로 진짜 손을 검출하는 성능은 미검증입니다.** 이 문서를 작성한
  환경에는 카메라 하드웨어가 없어, `--image` 모드로 파이프라인(로드→추론→판정→
  렌더) 전체가 예외 없이 도는 것까지는 확인했지만, 카메라 앞에 실제 손을 들고
  검출·판정이 잘 되는지는 하드웨어가 있는 환경에서 직접 실행해 확인해야 합니다.
  "동작합니다"라고 단정하지 않습니다.

- **[해소됨] 거울 모드와 판정 가정의 충돌** — 초기 구현은 엄지를
  `handedness` + x좌표 부호로 판정해서, `app.py`가 `cv2.flip(frame, 1)`로
  좌우 반전한 뒤 검출하는 웹캠 모드와 전제가 어긋났습니다. 이후 엄지를
  **거리비**(엄지끝↔검지밑동 ÷ 손목↔중지밑동)로 바꾸면서 `fingers_up()`이
  `handedness`를 아예 참조하지 않게 되어, 이 충돌은 **문제 자체가 사라졌습니다**.
  부호를 뒤집어 맞추는 대신 몰라도 되는 설계로 바꾼 결과입니다.

- **손이 기울면 검지~소지 판정이 무너집니다.** `fingers_up()`의 검지~소지
  판정은 "손끝 y좌표 < 두 번째 관절 y좌표"로 펴짐을 판단하는데, 이는 손이 대략
  세워진 자세라는 전제입니다. 손을 수평으로 돌리면 y차이가 작아지거나 부호가
  뒤집혀 오판정될 수 있습니다 — 손 전체의 회전을 보정하는 로직은 없습니다
  (`gesture.py`의 `fingers_up` 독스트링에 명시).
  참고로 **손등이 보이는 경우는 검증됐습니다** — 회귀 테스트에 쓰는 실사진 6장이
  전부 손등이 보이는 사진이고, 6/6 통과합니다.

- **엄지 임계값 0.57은 표본 6장에서 나온 값입니다.** 실측 분포는 굽음 0.217·0.266,
  펴짐 0.865~1.058로 사이가 넓게 비어 있어(폭 0.598) 그 중앙을 잡았습니다.
  손 크기·나이·카메라 거리가 크게 다른 사용자군에서는 재보정이 필요할 수 있고,
  손을 카메라 쪽으로 기울이면 정규화 분모가 작아져 엄지가 과대평가됩니다.

- **화면에 표시되는 신뢰도는 "손 검출 신뢰도"가 아닙니다.** Tasks API는 별도의
  손 검출 점수를 제공하지 않아, `hand.score`는 사실 handedness(왼손/오른손)
  분류 확률입니다(`landmarker.py`의 `_to_hand_results` 주석). 화면에 큰 숫자가
  떠도 "이게 진짜 손일 확률"로 오독하면 안 됩니다.

---

## 테스트

```bash
venv/bin/python -m pytest tests/ -q
```

실행 결과: **80 passed, 1 skipped** (2026-07-29 실측, `feature/rps-game` 병합 후).

- `test_gesture.py` — 22개. 카메라·모델 없이 합성 좌표 + 실사진 6장으로 판정 로직
  100% 검증(`gesture.py`가 순수 함수이기 때문에 가능합니다).
- `test_landmarker.py` — 25 passed + 1 skipped. 모델 로드, 수명주기,
  VIDEO 모드 타임스탬프 제약, MediaPipe 원시 결과 → 계약 타입 변환을 검증합니다.
  건너뛴 1개는 이름 그대로입니다: `test_실제_손_검출_정확도는_이_환경에서_검증_불가`
  — 실제 손 검출 정확도는 웹캠/실제 손 사진이 있어야 확인 가능하므로, 이 저장소는
  "검증 안 됐다"를 숨기지 않고 테스트 이름 자체로 드러냅니다.
- `test_rps.py` — 33개. 승부 판정 전 조합(3×3+UNKNOWN), AI 수 결정성,
  제스처 흔들림에 대한 프레임 안정화 로직을 순수 로직으로 검증합니다.

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | mediapipe 1.0.0이 `mp.solutions.*` 레거시 API를 완전히 제거하고 Tasks API(`mediapipe.tasks.python.vision`)만 남김 | `HandLandmarker.create_from_options()` 기반 Tasks API로 작성(`landmarker.py` 참고). 오래된 MediaPipe 튜토리얼·AI가 처음 제안하는 코드는 대부분 구버전 API다 |
| 한글 텍스트가 `???`로 깨짐 | `cv2.putText`는 라틴 문자 전용 내장 폰트만 지원 | `hud.py`처럼 PIL(`ImageDraw`)로 그린 뒤 numpy BGR로 되돌리는 라운드트립 필요. 추가 의존성 없음 — Pillow는 mediapipe/opencv-python 설치 시 이미 딸려 온다 |
| 웹캠에서 랜드마크가 프레임마다 흔들리거나 손을 자꾸 놓침 | `HandLandmarkerOptions`에 `running_mode`를 지정하지 않으면 기본값인 IMAGE 모드로 동작 — 매 프레임을 독립 이미지로 취급해 무거운 팜 디텍터가 매번 다시 돎 | `running_mode=vision.RunningMode.VIDEO`를 명시(`landmarker.py`에 이미 반영). 대신 VIDEO 모드는 `timestamp_ms`가 반드시 단조 증가해야 한다는 제약이 붙는다(벽시계 대신 프레임 인덱스 기반 타임스탬프 권장, `app.py`의 `FRAME_INTERVAL_MS` 참고) |
| `bench.py`·실사진 테스트가 `yolo/rps/` 파일을 못 찾음 **[해소됨]** | 워크샵 루트를 `HERE.parent`처럼 **고정 깊이**로 계산하면, `.worktrees/docs/`같이 한 단계 깊은 곳에서 실행할 때 `.worktrees/`를 가리켜 `yolo/rps/`를 놓친다 | `_find_workshop_root()`가 조상 디렉토리를 위로 탐색하도록 수정됨(`bench.py` `6e02bbd`, `tests/test_gesture.py` `01fab48`). **테스트 쪽이 특히 위험했다** — 경로가 없으면 `skipif`가 걸려 실사진 회귀 6개가 조용히 사라지는데도 초록불이었다. 실패하는 테스트는 눈에 띄지만 **사라진 테스트는 안 띈다** |
| macOS에서 카메라가 안 열림 | 카메라 권한 미승인. `cap.isOpened()`가 `True`인데도 프레임이 계속 빈 값으로 온다(권한 거부가 예외를 던지지 않고 조용히 실패) | 시스템 설정 > 개인정보 보호 및 보안 > 카메라에서 터미널/IDE 허용 |

---

## 프로젝트 구조

```
mediapipe/hand_lite/
├── app.py                  진입점 (--image 정지 이미지 모드 / 웹캠 모드) — 손 추적만
├── rps_game.py              가위바위보 게임 진입점 — app.py + 승부 판정 + 점수판
├── doctor.py                 웹캠 진단 도구 (--no-camera로 카메라 없이도 1~2단계 재현 가능)
├── hand_lite/
│   ├── types.py             계약: Landmark, HandResult, Gesture, HAND_CONNECTIONS
│   ├── landmarker.py        엔진: MediaPipe HandLandmarker 래퍼 (VIDEO 모드)
│   ├── gesture.py           판정: 순수 함수, 21점 좌표 기하학 if문
│   ├── rps.py                가위바위보 규칙: judge·MoveStabilizer(제스처 흔들림 보정)·Score
│   └── hud.py               렌더: OpenCV + PIL(한글 라운드트립)
├── bench.py                 경량성 벤치마크 (모델 용량 [사실] + 추론 속도 [추정])
├── tests/                   80 passed + 1 skipped (2026-07-29 실측)
├── docs/                     prd.md · architecture.md · progress.md
├── TUTORIAL.md                학생용 단계별 실습 튜토리얼
├── WEBCAM_VERIFY.md           웹캠 실검증 체크리스트 (사람이 직접 확인)
└── models/hand_landmarker.task   7.5MB, 저장소에 포함 — 별도 다운로드 불필요
```
