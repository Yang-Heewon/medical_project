"""Generator factory."""

from __future__ import annotations

from vision_rag_cxr.models.medgemma_generator import MedGemmaGenerator


def build_generator(config: dict):
    name = config.get("model_name", "").lower()
    if "medgemma" in name or "medgemma" in config.get("model_name_or_path", "").lower():
        return MedGemmaGenerator(config)
    raise ValueError(f"지원하지 않는 generator입니다: {name}")
