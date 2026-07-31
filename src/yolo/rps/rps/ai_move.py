"""AI가 낼 패를 Gemini 이미지 생성으로 준비한다.

시작 시 가위/바위/보 각 1장을 생성해 assets/ai_moves/ 에 캐싱하고,
라운드마다 그 중 하나를 무작위로 고른다. (승패 판정은 judge.py의 순수 코드가 담당)
"""

from __future__ import annotations

import os
import random
from pathlib import Path

from .judge import Move

IMAGE_MODEL = "gemini-3.1-flash-image"

_STYLE = (
    "flat vector illustration, bold black outlines, vibrant blue and orange colors, "
    "plain dark navy background, centered, square composition, no text, no watermark"
)

_PROMPTS: dict[Move, str] = {
    Move.ROCK: f"A friendly cartoon robot hand making a ROCK gesture — a closed fist, front view. {_STYLE}",
    Move.PAPER: f"A friendly cartoon robot hand making a PAPER gesture — an open palm with all five fingers "
                f"extended and spread, front view. {_STYLE}",
    Move.SCISSORS: f"A friendly cartoon robot hand making a SCISSORS gesture — only the index and middle "
                   f"fingers extended in a V shape, the other fingers folded, front view. {_STYLE}",
}


class AIMoveProvider:
    """AI 패 이미지 공급자.

    images가 비어 있으면 호출부는 이모지 폴백으로 전환한다.
    """

    def __init__(self, images: dict[Move, Path] | None = None) -> None:
        self.images: dict[Move, Path] = images or {}

    @property
    def has_images(self) -> bool:
        return len(self.images) == len(Move)

    def pick(self) -> Move:
        return random.choice(list(Move))

    def image_path(self, move: Move) -> Path | None:
        return self.images.get(move)


def _load_api_key(project_root: Path) -> str | None:
    """.env 파일 또는 시스템 환경변수에서 GEMINI_API_KEY를 읽는다."""
    try:
        from dotenv import load_dotenv

        load_dotenv(project_root / ".env")
    except ImportError:
        pass
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key or None


def _generate_one(client, move: Move, dest: Path) -> None:
    """이미지 1장을 생성해 dest에 저장한다. 실패 시 예외를 그대로 올린다."""
    from google.genai import types

    response = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=_PROMPTS[move],
        config=types.GenerateContentConfig(response_modalities=["Image"]),
    )
    candidates = response.candidates or []
    if not candidates or not candidates[0].content:
        raise RuntimeError(f"Gemini 응답에 콘텐츠가 없습니다 (move={move.value})")

    for part in candidates[0].content.parts:
        inline = getattr(part, "inline_data", None)
        if inline and inline.data:
            dest.write_bytes(inline.data)
            return
    raise RuntimeError(f"Gemini 응답에 이미지가 없습니다 (move={move.value})")


def prepare_ai_moves(
    project_root: Path,
    assets_dir: Path,
    regenerate: bool = False,
) -> AIMoveProvider:
    """가위/바위/보 이미지를 준비한다.

    - 이미 캐시된 파일이 있으면 재사용 (regenerate=True면 무시하고 재생성)
    - GEMINI_API_KEY가 없거나 생성에 실패하면 빈 provider를 반환 → 이모지 폴백
    """
    assets_dir.mkdir(parents=True, exist_ok=True)
    paths = {m: assets_dir / f"{m.value}.png" for m in Move}

    if not regenerate and all(p.exists() for p in paths.values()):
        print(f"[AI 패] 캐시된 이미지 사용: {assets_dir}")
        return AIMoveProvider(paths)

    api_key = _load_api_key(project_root)
    if not api_key:
        print(
            "\n" + "=" * 68 + "\n"
            "[AI 패] GEMINI_API_KEY를 찾지 못했습니다 — 이미지 생성을 건너뜁니다.\n"
            "  AI의 패는 이모지(✊ ✋ ✌)로 표시됩니다. 게임 진행에는 문제 없습니다.\n"
            "\n"
            "  이미지로 보려면 아래 중 하나를 하세요:\n"
            f"    1) {project_root / '.env'} 파일에 GEMINI_API_KEY=발급받은_키 를 적기\n"
            "    2) export GEMINI_API_KEY=발급받은_키 실행 후 재시작\n"
            "  키 발급: https://aistudio.google.com/apikey\n"
            + "=" * 68 + "\n"
        )
        return AIMoveProvider()

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
    except Exception as exc:  # noqa: BLE001 — 원인을 그대로 표면화
        print(f"[AI 패] Gemini 클라이언트 초기화 실패 → 이모지 폴백. 원인: {exc}")
        return AIMoveProvider()

    generated: dict[Move, Path] = {}
    for move, dest in paths.items():
        if dest.exists() and not regenerate:
            generated[move] = dest
            continue
        print(f"[AI 패] 이미지 생성 중... {move.value}")
        try:
            _generate_one(client, move, dest)
            generated[move] = dest
        except Exception as exc:  # noqa: BLE001
            print(f"[AI 패] '{move.value}' 생성 실패 → 이모지 폴백으로 전환. 원인: {exc}")
            return AIMoveProvider()

    print(f"[AI 패] 이미지 {len(generated)}장 준비 완료: {assets_dir}")
    return AIMoveProvider(generated)
