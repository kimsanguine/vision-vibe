"""AI Hub 재활용품 분류 데이터 서브샘플링 + YOLO 포맷 변환.

각 세부 카테고리(15종)에서 이미지 용량 합이 TARGET_BYTES(기본 100MB)를
넘지 않는 선에서 무작위로 뽑아, AI Hub의 JSON 바운딩박스 라벨을
YOLO txt 포맷(class cx cy w h, 0~1 정규화)으로 변환해 저장한다.
zip에서 직접 읽어 디스크에 압축 해제본을 남기지 않는다.
"""

import json
import random
import zipfile
from pathlib import Path

random.seed(42)

RAW = Path(__file__).parent.parent / "data" / "raw" / "232.재활용품_분류_및_선별_데이터" / "01-1.정식개방데이터" / "Validation"
SRC_DIR = RAW / "01.원천데이터"
LABEL_DIR = RAW / "02.라벨링데이터"
OUT = Path(__file__).parent.parent / "data" / "yolo_dataset"
TARGET_BYTES = 100 * 1024 * 1024  # 카테고리당 100MB

CATEGORIES = [
    ("VS_1.영상추출_01.금속캔_001.철캔.zip", "VL_1.영상추출_01.금속캔_001.철캔.zip"),
    ("VS_1.영상추출_01.금속캔_002.알루미늄캔.zip", "VL_1.영상추출_01.금속캔_002.알루미늄캔.zip"),
    ("VS_1.영상추출_02.종이_001.종이.zip", "VL_1.영상추출_02.종이_001.종이.zip"),
    ("VS_1.영상추출_03.페트병_001.무색단일.zip", "VL_1.영상추출_03.페트병_001.무색단일.zip"),
    ("VS_1.영상추출_03.페트병_002.유색단일.zip", "VL_1.영상추출_03.페트병_002.유색단일.zip"),
    ("VS_1.영상추출_04.플라스틱_001.PE.zip", "VL_1.영상추출_04.플라스틱_001.PE.zip"),
    ("VS_1.영상추출_04.플라스틱_002.PP.zip", "VL_1.영상추출_04.플라스틱_002.PP.zip"),
    ("VS_1.영상추출_04.플라스틱_003.PS.zip", "VL_1.영상추출_04.플라스틱_003.PS.zip"),
    ("VS_1.영상추출_05.스티로폼_001.스티로폼.zip", "VL_1.영상추출_05.스티로폼_001.스티로폼.zip"),
    ("VS_1.영상추출_06.비닐_001.비닐.zip", "VL_1.영상추출_06.비닐_001.비닐.zip"),
    ("VS_1.영상추출_07.유리병_001.갈색.zip", "VL_1.영상추출_07.유리병_001.갈색.zip"),
    ("VS_1.영상추출_07.유리병_002.녹색.zip", "VL_1.영상추출_07.유리병_002.녹색.zip"),
    ("VS_1.영상추출_07.유리병_003.투명.zip", "VL_1.영상추출_07.유리병_003.투명.zip"),
    ("VS_2.직접촬영_08.건전지_001.건전지.zip", "VL_2.직접촬영_08.건전지_001.건전지.zip"),
    ("VS_2.직접촬영_09.형광등_001.형광등.zip", "VL_2.직접촬영_09.형광등_001.형광등.zip"),
]

class_names: list[str] = []
class_index: dict[tuple[str, str], int] = {}


def get_class_id(cls: str, details: str) -> int:
    key = (cls, details)
    if key not in class_index:
        class_index[key] = len(class_names)
        class_names.append(f"{cls}_{details}")
    return class_index[key]


def main() -> None:
    img_out_dir = OUT / "images"
    lbl_out_dir = OUT / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for img_zip_name, lbl_zip_name in CATEGORIES:
        cat_label = (
            img_zip_name.replace("VS_1.영상추출_", "")
            .replace("VS_2.직접촬영_", "")
            .replace(".zip", "")
        )
        img_zip_path = SRC_DIR / img_zip_name
        lbl_zip_path = LABEL_DIR / lbl_zip_name

        with zipfile.ZipFile(img_zip_path) as zf:
            img_infos = [
                i for i in zf.infolist()
                if i.filename.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        random.shuffle(img_infos)

        with zipfile.ZipFile(lbl_zip_path) as zf:
            label_lookup = {Path(n).name: n for n in zf.namelist() if n.endswith(".json")}

        selected = []
        total = 0
        for info in img_infos:
            if total >= TARGET_BYTES:
                break
            base = Path(info.filename).name
            label_name = Path(base).stem + ".json"
            matching_label = label_lookup.get(label_name)
            if not matching_label:
                continue
            selected.append((info, matching_label))
            total += info.file_size

        with zipfile.ZipFile(img_zip_path) as izf, zipfile.ZipFile(lbl_zip_path) as lzf:
            for info, label_name in selected:
                ann = json.loads(lzf.read(label_name))
                iw = ann["IMAGE_INFO"]["IMAGE_WIDTH"]
                ih = ann["IMAGE_INFO"]["IMAGE_HEIGHT"]

                yolo_lines = []
                for a in ann.get("ANNOTATION_INFO", []):
                    points = a.get("POINTS")
                    shape_type = a.get("SHAPE_TYPE")
                    if not points:
                        continue
                    if shape_type == "BOX":
                        x, y, w, h = points[0]
                    elif shape_type == "POLYGON":
                        # 폴리곤 꼭짓점 [[x1,y1],[x2,y2],...] -> 외접 바운딩박스로 축소.
                        # 형태 정밀도는 잃지만 객체 탐지(YOLO)에는 박스만 있으면 된다.
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        x, y = min(xs), min(ys)
                        w, h = max(xs) - x, max(ys) - y
                    else:
                        continue
                    cls_id = get_class_id(a["CLASS"], a["DETAILS"])
                    cx = max(0.0, min(1.0, (x + w / 2) / iw))
                    cy = max(0.0, min(1.0, (y + h / 2) / ih))
                    nw = max(0.0, min(1.0, w / iw))
                    nh = max(0.0, min(1.0, h / ih))
                    yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                base = Path(info.filename).name
                stem = Path(base).stem
                ext = Path(base).suffix
                (img_out_dir / f"{stem}{ext}").write_bytes(izf.read(info))
                (lbl_out_dir / f"{stem}.txt").write_text("\n".join(yolo_lines))

        print(f"{cat_label}: {len(selected)}장, {total/1e6:.1f}MB 선택")
        summary.append((cat_label, len(selected), total / 1e6))

    (OUT / "classes.txt").write_text("\n".join(class_names))

    print("\n=== 클래스 목록 ===")
    for i, name in enumerate(class_names):
        print(i, name)

    print("\n=== 카테고리별 요약 ===")
    total_imgs = sum(s[1] for s in summary)
    total_mb = sum(s[2] for s in summary)
    for cat, n, mb in summary:
        print(f"{cat}: {n}장 ({mb:.1f}MB)")
    print(f"\n합계: {total_imgs}장, {total_mb:.1f}MB")


if __name__ == "__main__":
    main()
