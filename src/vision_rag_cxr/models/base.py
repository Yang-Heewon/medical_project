"""모델 abstract interface.

이 프레임워크의 핵심은 모델 교체 가능성이다.
MedGemma, Qwen-VL, LLaVA-Med 등 어떤 generator를 쓰더라도
실험 코드는 동일한 interface만 호출해야 한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseGenerator(ABC):
    """이미지 + 프롬프트를 받아 report 또는 localization JSON을 생성하는 VLM interface."""

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get("model_name", config.get("model_name_or_path", "unknown_generator"))

    @abstractmethod
    def generate_impression(self, sample: dict, prompt: str, context_examples: list[dict] | None = None) -> str:
        pass

    @abstractmethod
    def generate_localization(self, sample: dict, prompt: str, context_examples: list[dict] | None = None) -> dict:
        pass


class BaseVisionEncoder(ABC):
    """RAG DB 구축용 image/text encoder interface."""

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get("vision_encoder_name", config.get("model_name", "unknown_encoder"))

    @abstractmethod
    def encode_image(self, frontal_path: str, lateral_path: str | None = None) -> np.ndarray:
        pass

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        pass

    def encode_sample(self, sample: dict) -> dict[str, np.ndarray]:
        """sample 하나에 대해 image/text embedding을 모두 만든다."""
        return {
            "image_embedding": self.encode_image(sample["frontal_path"], sample.get("lateral_path")),
            "text_embedding": self.encode_text(sample.get("impression", "")),
        }


class BaseCritic(ABC):
    """TextGrad critic / feedback model interface."""

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get("model_name", config.get("model_name_or_path", "unknown_critic"))

    @abstractmethod
    def critique(self, prediction: str, target: str, sample: dict, metrics: dict) -> str:
        pass

    @abstractmethod
    def rewrite_style_profile(self, current_style_profile: str, critiques: list[str], metric_summary: dict) -> str:
        pass
