"""subsample.py가 만든 평탄한 images/labels를 YOLO 표준 train/val 구조로 분할."""

import random
import shutil
from pathlib import Path

random.seed(42)
VAL_RATIO = 0.2

BASE = Path(__file__).parent.parent / "data" / "yolo_dataset"
IMG_SRC = BASE / "images"
LBL_SRC = BASE / "labels"


def main() -> None:
    stems = sorted(p.stem for p in IMG_SRC.iterdir() if p.is_file())
    random.shuffle(stems)
    n_val = int(len(stems) * VAL_RATIO)
    val_stems = set(stems[:n_val])

    for split in ("train", "val"):
        (BASE / "images" / split).mkdir(parents=True, exist_ok=True)
        (BASE / "labels" / split).mkdir(parents=True, exist_ok=True)

    moved = {"train": 0, "val": 0}
    for img_path in list(IMG_SRC.iterdir()):
        if not img_path.is_file():
            continue
        stem = img_path.stem
        split = "val" if stem in val_stems else "train"
        lbl_path = LBL_SRC / f"{stem}.txt"
        shutil.move(str(img_path), BASE / "images" / split / img_path.name)
        if lbl_path.exists():
            shutil.move(str(lbl_path), BASE / "labels" / split / lbl_path.name)
        moved[split] += 1

    classes = (BASE / "classes.txt").read_text().splitlines()
    data_yaml = BASE / "data.yaml"
    data_yaml.write_text(
        "path: " + str(BASE.resolve()) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(classes)}\n"
        "names:\n" + "\n".join(f"  {i}: {name}" for i, name in enumerate(classes)) + "\n"
    )

    print(f"train: {moved['train']}장, val: {moved['val']}장")
    print(f"data.yaml 작성 완료: {data_yaml}")


if __name__ == "__main__":
    main()
