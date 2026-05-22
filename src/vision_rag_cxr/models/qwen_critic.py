"""Qwen3.5-9B critic adapter.

이 모델은 TextGrad에서 prompt/style profile을 평가하고 개선안을 제시하는 critic 역할이다.
VLM generator와 critic을 분리하면, generator는 MedGemma로 유지하면서 feedback model만 교체할 수 있다.
"""

from __future__ import annotations

from vision_rag_cxr.models.base import BaseCritic


class QwenCritic(BaseCritic):
    """Qwen 계열 critic model wrapper."""

    def critique(self, prediction: str, target: str, sample: dict, metrics: dict) -> str:
        # 실제 production에서는 transformers/vLLM/SGLang client 호출로 교체한다.
        return (
            "Prediction과 target의 임상적 차이를 분석하세요. "
            "누락된 병변, hallucination, normal collapse 가능성을 중심으로 prompt 개선 피드백을 생성하세요."
        )

    def rewrite_style_profile(self, current_style_profile: str, critiques: list[str], metric_summary: dict) -> str:
        # 실제 production에서는 Qwen에게 current_style_profile + critiques를 넣고 개선된 profile을 생성하게 한다.
        return current_style_profile + "\nAvoid unsupported normal conclusions and explicitly mention visible abnormal findings."
