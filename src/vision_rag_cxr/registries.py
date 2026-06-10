"""중앙 plug-in/plug-out 레지스트리.

dataset / generator / encoder / critic을 이름으로 끼우고 빼는 단일 진입 카탈로그.
- build_* 팩토리를 한 곳에서 재노출 (각 하위 패키지에 실제 구현)
- *_CATALOG dict: CLI(`vrag list`)와 sweep이 공유하는 사람이 읽는 카탈로그/preset

새 모델/데이터셋을 붙이려면: 해당 하위 패키지에 adapter를 추가하고 factory 분기 + 아래 카탈로그에 한 줄.
"""

from __future__ import annotations

# 팩토리 재노출 (plug-in/out 핵심)
from vision_rag_cxr.datasets.registry import PREPROCESSORS, preprocess_dataset
from vision_rag_cxr.models.critics.qwen import build_critic
from vision_rag_cxr.models.encoders.base import build_vision_encoder
from vision_rag_cxr.models.generators.factory import build_generator

# ---- ① 데이터셋 카탈로그 ----------------------------------------------------
DATASET_CATALOG = {
    "indiana": "로컬 Indiana CSV+이미지 (datasets.indiana.preprocess_indiana)",
    "indiana_hf": "HF ykumards/open-i 스트리밍 실제 IU (무인증, chest)",
    "roco": "HF eltorio/ROCOv2-radiology 스트리밍 (무인증, 멀티-모달리티, caption=impression, 라벨없음/텍스트전용)",
    "padchest_gr": "PadChest-GR — 24-label + bbox GT (BIMCV 승인 필요)",
}

# ---- ③ generator preset (backend transformers, V100 fp16, 이미지 다운스케일) ----
GENERATOR_CATALOG = {
    "qwen2.5-vl": {"model_name": "qwen2.5-vl", "model_name_or_path": "Qwen/Qwen2.5-VL-7B-Instruct",
                   "backend": "transformers", "device_map": "auto", "dtype": "float16",
                   "max_new_tokens": 256, "temperature": 0.0, "max_image_size": 896},
    "medgemma": {"model_name": "medgemma", "model_name_or_path": "google/medgemma-4b-it",
                 "backend": "transformers", "device_map": "auto", "dtype": "float16",
                 "max_new_tokens": 256, "temperature": 0.0, "max_image_size": 896},
    "llava-med": {"model_name": "llava-med", "model_name_or_path": "microsoft/llava-med-v1.5-mistral-7b",
                  "backend": "transformers", "device_map": "auto", "dtype": "float16",
                  "trust_remote_code": True, "max_new_tokens": 256, "temperature": 0.0, "max_image_size": 896},
    "placeholder": {"model_name": "medgemma", "model_name_or_path": "google/medgemma-4b-it", "backend": "placeholder"},
}

# ---- ③ encoder preset --------------------------------------------------------
ENCODER_CATALOG = {
    "biomedclip": {"vision_encoder_name": "medclip", "benchmark_tag": "biomedclip",
                   "model_name_or_path": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                   "device": "cuda", "batch_size": 16, "embedding_dim": 512},
    "dummy_hash": {"vision_encoder_name": "dummy_hash", "benchmark_tag": "dummy_hash", "embedding_dim": 384},
}

# 현재 제약/주의 (plug-in이지만 추가 작업 필요한 것)
CATALOG_NOTES = {
    "padchest_gr": "데이터 BIMCV 승인 필요(현재 미배치)",
    "llava-med": "model_type 'llava_mistral' — 표준 transformers 로드 불가(별도 LLaVA 코드 필요)",
}

__all__ = [
    "preprocess_dataset", "build_generator", "build_vision_encoder", "build_critic", "PREPROCESSORS",
    "DATASET_CATALOG", "GENERATOR_CATALOG", "ENCODER_CATALOG", "CATALOG_NOTES",
]
