"""로깅 유틸리티."""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """콘솔과 파일에 동시에 남길 수 있는 logger를 만든다."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s - %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
