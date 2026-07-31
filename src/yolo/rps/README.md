# YOLO 가위바위보 — 웹캠 vs 생성형 AI

웹캠에 낸 손모양을 YOLO로 인식하고, AI는 자기 패를 **Gemini로 생성한 이미지**로 보여준 뒤
실시간으로 승부를 판정하는 데스크톱 앱입니다.
"OpenCV/YOLO 비전 앱, 바이브 코딩으로 만들기" 워크샵의 확장 과제(Part 2·3에 이어지는 YOLO 편)입니다.

```
┌─ 당신 (웹캠 + YOLO 검출) ─┐  ┌─ AI (Gemini 생성 이미지) ─┐
│  [실시간 영상 + 박스]      │  │  [✊ / ✋ / ✌ 일러스트]    │
└───────────────────────────┘  └───────────────────────────┘
              승리!  ·  당신 보  vs  AI 바위
```

---

## 빠른 시작

```bash
cd 260728_vision_vibe_workshop/src/yolo/rps

python3 -m venv venv
venv/bin/pip install -r requirements.txt

# 모델 가중치 내려받기 (114MB — 저장소에 포함되지 않음)
venv/bin/python scripts/fetch_assets.py

# (선택) AI 패를 이미지로 보려면 API 키 등록
cp .env.example .env && $EDITOR .env      # GEMINI_API_KEY=... 채우기

venv/bin/python main.py
```

조작: **SPACE** 라운드 시작 · **R** 점수 초기화 · **Q / ESC** 종료

### 실행 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--source` | `0` | 카메라 인덱스 또는 영상 파일 경로 |
| `--model` | `models/rps_yolo11x_leeyunjai.pt` | YOLO 가중치 |
| `--device` | 자동 | `mps` / `cuda` / `cpu` |
| `--imgsz` | `640` | 추론 입력 크기 |
| `--conf` | `0.5` | 검출 신뢰도 임계값 |
| `--regen-ai-images` | — | AI 패 이미지 캐시 무시하고 재생성 |

웹캠 없이 시연하려면 테스트 영상을 소스로 넣으면 됩니다.

```bash
venv/bin/python scripts/fetch_assets.py --fixtures
venv/bin/python main.py --source tests/fixtures/rps_video.avi
```

---

## 동작 방식

| 단계 | 담당 | 방식 |
|---|---|---|
| 손모양 인식 | `rps/detector.py` | YOLO11x 객체 검출 → 최고 신뢰도 박스 1개 |
| 흔들림 제거 | `rps/detector.py` `MoveStabilizer` | 최근 7프레임 다수결(4표 이상) |
| AI의 패 | `rps/ai_move.py` | 시작 시 Gemini로 가위/바위/보 각 1장 생성 → 캐싱 → 라운드마다 무작위 선택 |
| 승패 판정 | `rps/judge.py` | **순수 if 로직** — LLM에 묻지 않음 |
| 라운드 진행 | `rps/game.py` | READY → COUNTDOWN(3초) → RESULT(2.5초) → READY |
| 화면 | `rps/hud.py` | PIL로 한글·이모지 렌더 후 `cv2.imshow` |

`cv2.putText`는 한글 글리프가 없어 깨지므로, 캔버스를 PIL로 그린 뒤 BGR로 되돌려 표시합니다.

### AI 패 이미지와 폴백

시작 시 `assets/ai_moves/{rock,paper,scissors}.png`를 확인하고, 없으면 Gemini
(`gemini-3.1-flash-image`)로 3장을 생성해 캐싱합니다. 이후 실행에서는 캐시를 재사용합니다.

`GEMINI_API_KEY`가 없거나 생성에 실패하면 **조용히 넘어가지 않고** 원인과 해결 방법을
콘솔에 출력한 뒤 정적 이모지(✊ ✋ ✌) 폴백으로 전환합니다. 게임 진행에는 지장이 없습니다.

---

## 모델 출처

가중치 후보 2개를 받아 실측 비교한 뒤 하나를 선택했습니다.

| 모델 | 출처 | 학습 | 저자 보고 mAP50 | 우리 실측 | 채택 |
|---|---|---|---|---|---|
| `rps_yolo11x_leeyunjai.pt` | [leeyunjai/yolo11-rps](https://huggingface.co/leeyunjai/yolo11-rps) (`rps_11x.pt`) | yolo11x, 500 epochs, Roboflow *Rock Paper Scissors SXSW v14* | **0.963** (P 0.953 / R 0.939) | 영상 20프레임 중 대부분 검출, 신뢰도 0.58~0.94 | ✅ |
| `rps_yolo11n_gholamreza.pt` | [Gholamreza/yolo11_rock_paper_scissors_detection](https://huggingface.co/Gholamreza/yolo11_rock_paper_scissors_detection) | yolo11n, **3 epochs** | **0.026** (P 0.0018) | 정지 이미지 6장·영상 13프레임 **전부 0건 검출** | ❌ |

두 모델 모두 `ultralytics.YOLO()`로 로드되고 클래스도 `{0:'Paper', 1:'Rock', 2:'Scissors'}` 3종으로
동일합니다. **로드 성공과 클래스 개수만으로는 사용 가능 여부를 판단할 수 없습니다** —
`yolo11n` 쪽은 3 epoch만 돌린 토이 체크포인트라 실제로는 아무것도 검출하지 못합니다.
체크포인트 내부의 `train_metrics`를 확인하는 것이 1차 판별법입니다.

클래스 인덱스 순서는 모델마다 다를 수 있으므로, 코드는 인덱스가 아니라 **클래스 이름**으로
매핑합니다(`rps/judge.py`의 `move_from_class_name`).

라이선스: 두 가중치 모두 Ultralytics 계보의 **AGPL-3.0**입니다.

---

## 성능

Apple Silicon(macOS) 실측. `--device`를 지정하지 않으면 `mps`가 자동 선택됩니다.

**추론만** (yolo11x, 영상 13프레임 평균)

| 디바이스 | imgsz | 속도 | 검출률 |
|---|---|---|---|
| cpu | 640 | 243 ms (4.1 FPS) | 10/13 |
| **mps** | **640** | **59 ms (17.0 FPS)** | **10/13** |
| mps | 480 | 37 ms (26.7 FPS) | 8/13 |
| mps | 320 | 24 ms (41.1 FPS) | 4/13 |

`imgsz`를 낮추면 빨라지지만 검출률이 급격히 떨어집니다. **mps + 640이 최적**입니다.

**전체 루프** (캡처 + 추론 + HUD, 웹캠 실측)

| 캡처 해상도 | 캡처 | 추론 | HUD | 합계 |
|---|---|---|---|---|
| 1920×1080 | 10.7 ms | 110.9 ms | 7.7 ms | 129 ms (7.7 FPS) |
| **1280×720** | 12.0 ms | 52.0 ms | 5.1 ms | **69 ms (14.5 FPS)** |
| 640×480 | 22.1 ms | 65.6 ms | 7.0 ms | 95 ms (10.6 FPS) |

병목은 캡처가 아니라 추론 전처리입니다. ultralytics가 원본 프레임을 640 정사각으로
letterbox하는 비용이 입력 해상도에 비례하기 때문에, 카메라를 720p로 고정하면
1080p 대비 약 1.9배 빨라집니다. `main.py`의 `CAPTURE_W/H`가 이 값을 지정합니다.
(640×480이 더 느린 것은 카메라가 그 해상도를 네이티브 지원하지 않아 변환 비용이 붙기 때문입니다.)

---

## 테스트

```bash
venv/bin/python -m pytest tests/ -q
```

`tests/test_detector.py`의 영상 기반 테스트는 픽스처가 없으면 자동으로 skip됩니다.
전부 실행하려면 `venv/bin/python scripts/fetch_assets.py --fixtures`를 먼저 돌리세요.

---

## 검증 상태

정직하게 구분합니다.

### 검증됨

| 항목 | 방법 | 결과 |
|---|---|---|
| 모델 로드 · 클래스 확인 | 두 모델 모두 `YOLO()` 로드 후 `names` 확인 | 둘 다 로드 OK, 3종 클래스 확인 |
| 모델 선별 | 체크포인트 `train_metrics` + 정지 이미지 6장 + 영상 프레임 실측 | yolo11n 0건 검출 → 탈락, yolo11x 채택 |
| 모델 출처 | 재다운로드 후 SHA256 대조 | `ec362054…003e` 일치 — `leeyunjai/yolo11-rps` 확정 |
| `model.predict()` 무오류 | 정지 이미지 · 영상 프레임 · 빈 프레임 | 예외 없음, 손 없으면 `None` 반환 |
| 판정 로직 | 9가지 조합 전수 + 반대칭성 + 무승부 조건 | 37개 테스트 통과 |
| 상태 머신 | READY/COUNTDOWN/RESULT 전이, 인식 실패 시 점수 불변 | 통과 |
| Gemini 이미지 생성 | 실제 API 호출 3회 | 3장 생성 성공, 손모양 육안 확인 |
| 이모지 폴백 | `AIMoveProvider()` 빈 provider로 렌더 | ✊ 정상 표시 |
| HUD 렌더링 | 전체 루프를 헤드리스로 돌려 캔버스 PNG 덤프 | 한글·점수·결과 정상 |
| `cv2.imshow` | 윈도우 생성 후 파괴 | OK (COCOA 백엔드) |
| **웹캠 실캡처** | 실제 카메라로 60프레임 전 경로 실행 | 프레임 취득 → 추론 → HUD까지 동작 (720p 기준 14.5 FPS) |
| **앱 전 경로 (GUI)** | `main.py`를 영상 소스로 실행하고 창에 SPACE 2회 · `q` 전송 | 라운드 2회 진행·판정·정상 종료 → `최종 전적: 2승 0패 0무` |

### 검증되지 않음

- **사람이 실제로 손을 내밀었을 때의 인식 정확도.** 웹캠 루프는 동작을 확인했지만
  카메라 앞에 손을 내민 상태로는 검증하지 못했습니다(검출 0건 상태로만 확인).
  손모양 인식 자체는 녹화 영상으로 검증했으나, 조명·배경·거리에 따른 실사용 정확도는
  직접 손을 보여주며 확인해야 합니다.
- **검출률 수치의 독립성.** 테스트 영상 `rps_video.avi`는 원저자가 YOLO 추론 결과를
  **박스와 라벨이 화면에 구워진 상태로** 저장한 파일입니다(그래서 실행하면 파란 박스가 하나 더 보입니다).
  따라서 이 영상으로 잰 검출률은 참고치이지 독립적인 벤치마크가 아닙니다.
- **정지 이미지 픽스처의 대표성.** `tests/fixtures/*.png`는 CG로 렌더한 합성 이미지라
  학습 분포(실사 웹캠)와 다릅니다. 실제로 이 이미지들에서는 가위를 보로 오분류합니다.
  같은 손모양을 실사 영상에서는 0.92~0.94 신뢰도로 정확히 잡습니다.

---

## 프로젝트 구조

```
yolo/rps/
├── main.py                  앱 진입점 (캡처 루프 + 키 입력)
├── rps/
│   ├── judge.py             Move/Outcome 정의, 순수 판정 로직
│   ├── detector.py          YOLO 래퍼 + 프레임 다수결 안정화
│   ├── ai_move.py           Gemini 이미지 생성·캐싱·폴백
│   ├── game.py              라운드 상태 머신 + 점수
│   └── hud.py               PIL 기반 한글/이모지 화면 구성
├── scripts/fetch_assets.py  가중치·픽스처 내려받기
├── tests/                   37개 테스트
├── models/                  YOLO 가중치 (gitignore)
└── assets/ai_moves/         생성된 AI 패 이미지 (gitignore)
```

## 문제 해결

**카메라가 열리지 않음** — macOS 시스템 설정 > 개인정보 보호 및 보안 > 카메라에서
터미널 앱에 권한을 허용하고, 다른 앱이 카메라를 점유 중인지 확인하세요.

**인식이 잘 안 됨** — 손을 카메라에 가깝게, 배경과 대비되게 두세요. `--conf 0.35`로
임계값을 낮추면 검출은 늘지만 오인식도 늘어납니다.

**느림** — `--device` 없이 실행해 `mps` 자동 선택을 쓰세요. 그래도 느리면 `--imgsz 480`이
있지만 검출률 하락을 감수해야 합니다.

**AI 패가 이모지로만 나옴** — `.env`의 `GEMINI_API_KEY`를 확인하세요. 실행 시 콘솔에
원인이 출력됩니다.
