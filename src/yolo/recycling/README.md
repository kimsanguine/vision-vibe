# YOLO #3 — 재활용 분리수거 분류기 데이터 준비

AI Hub "재활용품 분류 및 선별 데이터"(datasetkey 71362, 전체 2,700GB)에서 15개
세부 카테고리 전부를 포함하되, 카테고리당 100MB로 제한해 서브샘플링한 YOLO
학습용 데이터셋을 만드는 파이프라인.

## 1. 데이터 출처

- AI Hub: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71362
- Validation 세트의 "1.영상추출"(01~07 카테고리, 500MB~2GB) + "2.직접촬영"(08~09
  건전지/형광등만, 영상추출 버전이 없어 이걸 사용, 각 1GB)을 원본으로 사용
  — Training 세트의 "2.직접촬영"(카테고리당 최대 97GB)은 논외

## 2. 실행 순서

```bash
# 1. .env에 AIHUB_API_KEY 설정 (절대 git에 커밋 금지 — .gitignore 처리됨)
# 2. aihubshell로 라벨+소스 다운로드 (scripts 참고, datasetkey=71362)
# 3. 서브샘플링 + YOLO 포맷 변환
python3 scripts/subsample.py
# 4. train/val 분할 + data.yaml 생성
python3 scripts/split_dataset.py
# 5. (다음 단계) YOLO 학습
# yolo detect train data=data/yolo_dataset/data.yaml model=yolo11n.pt epochs=50
```

## 3. 겪은 두 가지 버그 (직접 고침)

1. **aihubshell 자체 버그** — 공식 스크립트의 `merge_parts()`가 파일명을
   `printf '%q'`로 이스케이프해서 `find -name`에 넘기는데, `find`는 셸
   이스케이프를 해석하지 못해 매치 0건 → **빈 파일로 덮어쓰고 원본 part는
   삭제**하는 데이터 유실 버그가 있었다. 로컬 사본(`aihubshell`)에서
   `find` 없이 bash glob으로 직접 매칭하도록 고쳤다. Part 번호가 순번이
   아니라 바이트 오프셋(`part0`, `part1073741824`, ...)이라는 것도 이 과정에서
   확인했다.

2. **AI Hub 라벨 스키마 — BOX/POLYGON 혼재** — 애노테이션의 `SHAPE_TYPE`이
   전부 `BOX`([x,y,w,h])일 거라 가정하고 짰더니 전체의 약 30%가
   `POLYGON`(다각형 꼭짓점 리스트)이라 빈 라벨이 나왔다. 폴리곤 꼭짓점의
   외접 바운딩박스로 축소하는 변환을 추가해 해결(`scripts/subsample.py`).

## 4. 최종 데이터셋

| 항목 | 값 |
|---|---|
| 클래스 수 | 15 (`data/yolo_dataset/classes.txt`) |
| 총 이미지 | 2,419장 (train 1,936 / val 483) |
| 총 바운딩박스 | 4,817개, 전부 정규화 좌표 [0,1] 범위 검증 통과 |
| 용량 | 원본 다운로드 18GB → 서브샘플링 후 1.5GB (**원본은 삭제 완료**) |
| 검증 방식 | 임의 4장에 바운딩박스를 그려 실제 객체(병뚜껑·캔·유리병 등) 위치와 일치하는지 육안 확인 |

## 5. 알아둘 점

- 이미지는 재활용 선별장 컨베이어벨트 장면이라, 한 이미지 안에 여러 잡다한
  쓰레기가 같이 찍혀있다. **라벨링된 건 그 순간 선별 대상인 카테고리 물체
  뿐**이고, 나머지 잡다한 물체는 박스가 없다 — 완전한 다중 클래스 탐지가
  아니라 "이 이미지의 주 대상"에 가깝다는 걸 학생들에게 미리 알려줄 필요가
  있다.
- 건전지(31장)·형광등(26장)은 다른 카테고리(160~200장)보다 이미지 수가
  훨씬 적다 — 개별 사진 파일 용량이 커서(직접촬영본이라) 100MB 제한에
  더 빨리 도달하기 때문. 클래스 불균형을 학습 시 고려해야 한다.
