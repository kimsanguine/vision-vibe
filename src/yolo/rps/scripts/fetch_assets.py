"""모델 가중치와 테스트 픽스처를 Hugging Face에서 내려받는다.

가중치(114MB)와 테스트 영상(36MB)은 저장소에 넣지 않으므로 이 스크립트로 받는다.

    venv/bin/python scripts/fetch_assets.py            # 모델만
    venv/bin/python scripts/fetch_assets.py --fixtures # 테스트 픽스처까지
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".hf_cache"

# (repo_id, repo_type, 원본 파일명, 저장 경로)
MODELS = [
    ("leeyunjai/yolo11-rps", "model", "rps_11x.pt", ROOT / "models" / "rps_yolo11x_leeyunjai.pt"),
]

FIXTURES = [
    ("Gholamreza/yolo11_rock_paper_scissors_detection", "model", "rps_video.avi",
     ROOT / "tests" / "fixtures" / "rps_video.avi"),
] + [
    ("ikarosdev/Rock-Paper-Scissors", "dataset", f"test/{cls}/test{cls}01-0{i}.png",
     ROOT / "tests" / "fixtures" / f"{cls}_{i}.png")
    for cls in ("paper", "rock", "scissors")
    for i in (0, 1)
]


def fetch(items) -> None:
    for repo_id, repo_type, filename, dest in items:
        if dest.exists():
            print(f"  건너뜀 (이미 있음): {dest.relative_to(ROOT)}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  받는 중: {repo_id}/{filename}")
        src = hf_hub_download(repo_id, filename, repo_type=repo_type, cache_dir=str(CACHE))
        shutil.copy(src, dest)
        print(f"    → {dest.relative_to(ROOT)} ({dest.stat().st_size:,} bytes)")


def main() -> None:
    p = argparse.ArgumentParser(description="모델·픽스처 내려받기")
    p.add_argument("--fixtures", action="store_true", help="테스트 픽스처도 함께 받는다")
    args = p.parse_args()

    print("[모델]")
    fetch(MODELS)
    if args.fixtures:
        print("[테스트 픽스처]")
        fetch(FIXTURES)
    print("완료.")


if __name__ == "__main__":
    main()
