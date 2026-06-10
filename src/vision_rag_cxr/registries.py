"""중앙 plug-in/plug-out 레지스트리.

dataset / generator / encoder / critic을 이름으로 끼우고 빼는 단일 진입 카탈로그.
- build_* 팩토리를 한 곳에서 재노출 (각 하위 패키지에 실제 구현)
- *_CATALOG dict: CLI(`vrag list`)와 sweep이 공유하는 사람이 읽는 카탈로그/preset

새 모델/데이터셋을 붙이려면: 해당 하위 패키지에 adapter를 추가하고 factory 분기 + 아래 카탈로그에 한 줄.
"""

from __future__ import annotations

# 팩토리 재노출 (plug-in/out 핵심)
from vision_rag_cxr.datasets.registry import PREPROCESSORS, preprocess_dataset
from vision_rag_cxr.datasets.labeler_chexbert import LABELER_CATALOG, build_labeler
from vision_rag_cxr.models.critics.qwen import build_critic
from vision_rag_cxr.models.encoders.base import build_vision_encoder
from vision_rag_cxr.models.generators.factory import build_generator
from vision_rag_cxr.prompting.registry import PROMPT_CATALOG, build_style_profile

# ---- ① 데이터셋 카탈로그 ----------------------------------------------------
DATASET_CATALOG = {
    "indiana": "로컬 Indiana CSV+이미지 (datasets.indiana.preprocess_indiana)",
    "indiana_hf": "HF ykumards/open-i 스트리밍 실제 IU (무인증, chest)",
    "roco": "HF eltorio/ROCOv2-radiology 스트리밍 (무인증, 멀티-모달리티, caption=impression, 라벨없음/텍스트전용)",
    "padchest_gr": "PadChest-GR (arXiv:2411.05085) — grounded report: 24-label + bbox GT(정규화 xyxy). 실제 배포본 배치됨(b2drop)",
}

# ---- ③ generator preset (backend transformers, V100 fp16) ----
# 이미지 해상도: PIL 선축소 없이 VLM 프로세서의 네이티브 리사이즈에 위임(공정성: 같은 VLM이 받는 사이즈).
# Qwen2.5-VL은 max_pixels(28-grid 픽셀 예산)로 묶는다. 1,048,576px ≈ 1024² longest-edge,
# 토큰 ≈ max_pixels/784 ≈ 1337/이미지 → V100 32GB 안전, 896 선축소보다 작은 병변 보존에 유리.
# (모든 조건 No-RAG/Vision-RAG/oracle에 동일 적용되므로 비교는 공정.)
_QWEN_MAX_PIXELS = 1_048_576
GENERATOR_CATALOG = {
    "qwen2.5-vl": {"model_name": "qwen2.5-vl", "model_name_or_path": "Qwen/Qwen2.5-VL-7B-Instruct",
                   "backend": "transformers", "device_map": "auto", "dtype": "float16",
                   "max_new_tokens": 256, "temperature": 0.0, "max_pixels": _QWEN_MAX_PIXELS},
    "medgemma": {"model_name": "medgemma", "model_name_or_path": "google/medgemma-4b-it",
                 "backend": "transformers", "device_map": "auto", "dtype": "float16",
                 "max_new_tokens": 256, "temperature": 0.0},  # MedGemma(SigLIP)는 프로세서가 고정 해상도로 리사이즈
    "llava-med": {"model_name": "llava-med", "model_name_or_path": "microsoft/llava-med-v1.5-mistral-7b",
                  "backend": "transformers", "device_map": "auto", "dtype": "float16",
                  "trust_remote_code": True, "max_new_tokens": 256, "temperature": 0.0},
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
    "padchest_gr": "grounded_reports JSON+master_table로 canonical 구축 완료(4555 study, bbox GT 3008); 이미지 zip(38.5GB) 해제 후 inference",
    "llava-med": "model_type 'llava_mistral' — 표준 transformers 로드 불가(별도 LLaVA 코드 필요)",
}

__all__ = [
    "preprocess_dataset", "build_generator", "build_vision_encoder", "build_critic", "build_labeler",
    "build_style_profile", "PREPROCESSORS", "DATASET_CATALOG", "GENERATOR_CATALOG", "ENCODER_CATALOG",
    "LABELER_CATALOG", "PROMPT_CATALOG", "CATALOG_NOTES",
]
