"""입출력 유틸리티.

이 파일은 프로젝트 전체에서 YAML/JSONL/CSV/Parquet을 읽고 쓰는 기능을 담당한다.
실험 코드에서 경로 처리 로직이 흩어지면 재현성이 나빠지므로, 공통 입출력은 여기로 모은다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    """디렉토리가 없으면 생성하고 Path 객체를 반환한다."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    """YAML config를 dict로 읽는다."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """dict record list를 JSONL로 저장한다."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """JSONL 파일을 list[dict]로 읽는다."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """확장자에 맞춰 DataFrame을 저장한다."""
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"지원하지 않는 DataFrame 저장 확장자입니다: {path.suffix}")
