"""워크샵 실습용 합성 한국 번호판 샘플 이미지 생성.

실제 차량 사진은 저작권 리스크가 있어(plate/spec.md 8절 명시),
간단한 차량 형태 위에 번호판을 합성해 생성한다.
파일명 = 정답 텍스트 (plate_app.py 탭2 샘플 테스트가 이 규칙으로 채점한다).
"""

import os
import random
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "260326_streamlit_test", "plate", "samples",
)

PLATES = [
    ("123가4567", "white"),
    ("456나7890", "white"),
    ("789다1234", "white"),
    ("서울12가3456", "white"),
    ("부산34나5678", "white"),
    ("701마2345", "yellow"),
    ("234라6789", "white"),
    ("대구56다7890", "white"),
]

CANVAS_W, CANVAS_H = 800, 600


def draw_car_scene(plate_text: str, plate_color: str) -> Image.Image:
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), (150, 170, 185))  # 하늘색 배경
    draw = ImageDraw.Draw(img)

    # 바닥
    draw.rectangle([0, 480, CANVAS_W, CANVAS_H], fill=(90, 90, 90))

    # 차체 (범퍼 포함 사다리꼴)
    body_color = (40, 45, 55)
    draw.polygon(
        [(150, 460), (650, 460), (700, 300), (500, 200), (300, 200), (100, 300)],
        fill=body_color,
    )
    # 범퍼
    draw.rectangle([160, 400, 640, 470], fill=(25, 28, 34))

    # 헤드라이트
    draw.ellipse([180, 320, 260, 380], fill=(230, 230, 200))
    draw.ellipse([540, 320, 620, 380], fill=(230, 230, 200))

    # 번호판 배경
    plate_w, plate_h = 260, 80
    px1 = CANVAS_W // 2 - plate_w // 2
    py1 = 410
    px2, py2 = px1 + plate_w, py1 + plate_h
    bg = (255, 255, 255) if plate_color == "white" else (250, 210, 40)
    draw.rectangle([px1, py1, px2, py2], fill=bg, outline=(0, 0, 0), width=3)

    font_size = 44 if len(plate_text) <= 9 else 36
    font = ImageFont.truetype(FONT_PATH, font_size, index=0)
    bbox = draw.textbbox((0, 0), plate_text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = px1 + (plate_w - tw) // 2 - bbox[0]
    ty = py1 + (plate_h - th) // 2 - bbox[1]
    draw.text((tx, ty), plate_text, fill=(0, 0, 0), font=font)

    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(42)
    for text, color in PLATES:
        img = draw_car_scene(text, color)
        out_path = os.path.join(OUT_DIR, f"{text}.jpg")
        img.save(out_path, quality=92)
        print("생성:", out_path)


if __name__ == "__main__":
    main()
