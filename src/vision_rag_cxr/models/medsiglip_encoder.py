"""MedSigLIPEncoder adapter.

MedSigLIP은 image/text common embedding space 기반 semantic retrieval에 적합하다.
현재 파일은 plug-and-play 위치를 명확히 하기 위한 skeleton이다.
실제 모델 설치 후 encode_image/encode_text 구현만 채우면 RAG DB build 코드는 그대로 재사용된다.
"""

from __future__ import annotations

import numpy as np

from vision_rag_cxr.models.base import BaseVisionEncoder


class MedSigLIPEncoder(BaseVisionEncoder):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None
        self.processor = None

    def encode_image(self, frontal_path: str, lateral_path: str | None = None) -> np.ndarray:
        raise NotImplementedError("MedSigLIPEncoder.encode_image를 실제 모델 환경에 맞게 구현하세요.")

    def encode_text(self, text: str) -> np.ndarray:
        raise NotImplementedError("MedSigLIPEncoder.encode_text를 실제 모델 환경에 맞게 구현하세요.")
