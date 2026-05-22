"""VLM output parser.

VLM은 종종 JSON 앞뒤로 설명문을 붙이므로, 가능한 한 JSON 부분만 복구한다.
"""

from __future__ import annotations

import json
import re


def parse_json_output(text: str) -> tuple[dict | None, str | None]:
    """문자열에서 JSON object를 파싱한다.

    반환:
    - parsed dict 또는 None
    - error message 또는 None
    """
    if isinstance(text, dict):
        return text, None

    s = str(text).strip()
    try:
        return json.loads(s), None
    except Exception as e:
        first = s.find("{")
        last = s.rfind("}")
        if first >= 0 and last > first:
            try:
                return json.loads(s[first:last + 1]), None
            except Exception as e2:
                return None, str(e2)
        return None, str(e)
