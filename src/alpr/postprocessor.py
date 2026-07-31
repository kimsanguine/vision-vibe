"""번호판 텍스트 후처리 모듈."""

import re

# 지역명 목록
REGIONS = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
]

# 번호판 포맷 정규식
# 신형: 숫자3 + 한글1 + 숫자4 (예: 123가4567)
NEW_FORMAT = re.compile(r'^(\d{3})([가-힣])(\d{4})$')

# 구형: 지역명 + 숫자2~3 + 한글1 + 숫자4 (예: 서울12가3456)
OLD_FORMAT = re.compile(
    r'^(' + '|'.join(REGIONS) + r')(\d{2,3})([가-힣])(\d{4})$'
)

# OCR 오인식 보정 맵 (자주 혼동되는 문자)
CHAR_CORRECTIONS = {
    'O': '0', 'o': '0', 'Q': '0',
    'I': '1', 'l': '1', '|': '1',
    'Z': '2', 'z': '2',
    'S': '5', 's': '5',
    'B': '8', 'b': '8',
    'G': '6', 'g': '6',
}


def postprocess(raw_text: str) -> dict:
    """OCR 결과를 정리하고 번호판 포맷을 검증한다.

    Returns:
        {
            "raw": 원본 텍스트,
            "cleaned": 정리된 텍스트,
            "format": "new" | "old" | "unknown",
            "valid": True/False,
            "parts": {"region": ..., "number1": ..., "hangul": ..., "number2": ...}
        }
    """
    # 공백, 특수문자 제거
    cleaned = re.sub(r'[^0-9가-힣a-zA-Z]', '', raw_text)

    # 영문자 → 숫자 보정 (한글 사이의 영문자는 보정하지 않음)
    corrected = _correct_chars(cleaned)

    result = {
        "raw": raw_text,
        "cleaned": corrected,
        "format": "unknown",
        "valid": False,
        "parts": {},
    }

    # 신형 포맷 매칭
    m = NEW_FORMAT.match(corrected)
    if m:
        result["format"] = "new"
        result["valid"] = True
        result["parts"] = {
            "number1": m.group(1),
            "hangul": m.group(2),
            "number2": m.group(3),
        }
        return result

    # 구형 포맷 매칭
    m = OLD_FORMAT.match(corrected)
    if m:
        result["format"] = "old"
        result["valid"] = True
        result["parts"] = {
            "region": m.group(1),
            "number1": m.group(2),
            "hangul": m.group(3),
            "number2": m.group(4),
        }
        return result

    return result


def _correct_chars(text: str) -> str:
    """OCR 오인식 문자를 보정한다.

    한글 문자는 보정하지 않고,
    숫자 위치에 있는 영문자만 숫자로 변환한다.
    """
    result = []
    for ch in text:
        if '가' <= ch <= '힣':
            # 한글은 그대로
            result.append(ch)
        elif ch in CHAR_CORRECTIONS:
            result.append(CHAR_CORRECTIONS[ch])
        else:
            result.append(ch)
    return ''.join(result)


def format_plate_text(parts: dict, fmt: str) -> str:
    """번호판 parts를 읽기 좋은 형식으로 포맷한다."""
    if fmt == "new":
        return f"{parts['number1']}{parts['hangul']} {parts['number2']}"
    elif fmt == "old":
        return f"{parts['region']} {parts['number1']}{parts['hangul']} {parts['number2']}"
    return ""
