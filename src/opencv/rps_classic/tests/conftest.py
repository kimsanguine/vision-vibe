"""프로젝트 루트를 import 경로에 넣는다 — 패키지 설치 없이 pytest 를 돌리기 위해.

setup.py/pyproject 를 두지 않는 이유: 이 프로젝트는 워크샵 실습용 단일
디렉토리라 설치 절차를 요구하지 않는 편이 낫다(Rule 2 — 최소 코드).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
