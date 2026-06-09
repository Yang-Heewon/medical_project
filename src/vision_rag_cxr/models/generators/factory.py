"""Generator factory (registry).

새 VLM generator를 붙이려면 BaseGenerator 어댑터를 만들고 여기 등록하면 된다.
model_name 또는 model_name_or_path에서 키워드를 보고 적절한 어댑터를 고른다.
"""

from __future__ import annotations

from vision_rag_cxr.models.generators.hf_vlm import HFVLMGenerator, LlavaMedGenerator, Qwen25VLGenerator
from vision_rag_cxr.models.generators.medgemma import MedGemmaGenerator


def build_generator(config: dict):
    name = str(config.get("model_name", "")).lower()
    path = str(config.get("model_name_or_path", "")).lower()
    blob = f"{name} {path}"

    if "medgemma" in blob:
        return MedGemmaGenerator(config)
    if "qwen2.5-vl" in blob or "qwen2_5_vl" in blob or "qwen2-vl" in blob or "qwenvl" in blob or "qwen-vl" in blob:
        return Qwen25VLGenerator(config)
    if "llava-med" in blob or "llava_med" in blob or "llava" in blob:
        return LlavaMedGenerator(config)
    # model_name_or_path가 주어졌으면 generic HF image-text-to-text로 시도한다.
    if path.strip():
        return HFVLMGenerator(config)
    raise ValueError(f"지원하지 않는 generator입니다: {name or path}")
