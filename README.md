# vision-vibe

**라이브 데모**: https://kimsanguine.github.io/vision-vibe/

브라우저에서 OpenCV.js / MediaPipe / YOLO(ONNX)로 손 인식, 가위바위보, 사물 인식, 문서 스캔 등을 체험하는 Flutter Web 앱입니다.

## 저장소 구성

이 저장소는 두 종류의 콘텐츠가 함께 있습니다.

| 폴더/파일 | 용도 |
|---|---|
| `index.html`, `main.dart.js`, `flutter_bootstrap.js`, `canvaskit/`, `assets/`, `icons/`, `js/`, `manifest.json`, `version.json` | **웹앱 실행에 필요한 빌드 산출물** — GitHub Pages가 서빙하는 파일 |
| `lecture/` | 강의용 Jupyter 노트북(`01~07`)과 정답(`answers/`), 참고 PDF/이미지 — 웹앱 동작과 무관 |
| `src/` | 강의 실습 원본 소스(`alpr`, `mediapipe`, `opencv`, `yolo`, `scripts`) — 웹앱 동작과 무관 |

즉 `lecture/`, `src/`를 지우거나 옮겨도 배포된 웹앱(`index.html` 이하)은 그대로 동작합니다. 웹앱 코드를 수정하려면 이 저장소가 아니라 원본 Flutter 프로젝트(`part3_flutter_web_d`)에서 빌드한 뒤 그 결과물만 이 저장소에 반영합니다.
