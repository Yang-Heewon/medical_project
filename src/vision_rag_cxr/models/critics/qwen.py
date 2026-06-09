"""Qwen critic adapter for TextGrad-style prompt optimization.

이 모델은 TextGrad loop에서 generator(MedGemma) prediction을 GT와 비교해
(1) 임상적 critique를 생성하고, (2) STYLE_PROFILE prompt fragment를 개선한다.
generator weight는 절대 건드리지 않는다 (prompt variable만 최적화).

backend:
- ``placeholder``: 모델 없이도 loop가 끝까지 돌도록 하는 heuristic critic.
  metric_summary를 반영해 결정적으로 rewrite하므로 배선/게이트 검증에 쓸 수 있다.
- ``transformers``: 실제 Qwen causal LM을 로드해 critique/rewrite를 생성한다.
"""

from __future__ import annotations

from typing import Any

from vision_rag_cxr.models.base import BaseCritic


class QwenCritic(BaseCritic):
    """Qwen 계열 critic model wrapper."""

    def __init__(self, config: dict):
        super().__init__(config)
        # 기본은 placeholder. 실제 모델을 쓰려면 config에 backend: transformers를 명시한다.
        # serve_backend(transformers/vllm/sglang)는 backend가 real일 때의 serving 엔진 힌트다.
        self.backend = str(config.get("backend", "placeholder")).lower()
        self.model_name_or_path = config.get("model_name_or_path", "Qwen/Qwen3.5-9B")
        self._loaded = False
        self.model = None
        self.tokenizer = None
        self.torch = None

    # --- model loading -------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return
        if self.backend in {"placeholder", "dummy", "mock"}:
            self._loaded = True
            return
        if self.backend not in {"transformers", "hf"}:
            raise ValueError(f"지원하지 않는 critic backend입니다: {self.backend}")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        dtype = getattr(torch, str(self.config.get("dtype", "bfloat16")), torch.bfloat16)
        local_files_only = bool(self.config.get("local_files_only", False))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, local_files_only=local_files_only, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=dtype,
            device_map=self.config.get("device_map", "auto"),
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        self.model.eval()
        self._loaded = True

    def _chat(self, system: str, user: str) -> str:
        """transformers backend chat 호출."""
        self.load()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.config.get("max_new_tokens", 1024)),
            "do_sample": float(self.config.get("temperature", 0.2)) > 0.0,
            "temperature": float(self.config.get("temperature", 0.2)) or None,
            "top_p": float(self.config.get("top_p", 0.9)),
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # --- critic API ----------------------------------------------------------
    def critique(self, prediction: str, target: str, sample: dict, metrics: dict) -> str:
        if self.backend in {"placeholder", "dummy", "mock"}:
            missed = metrics.get("missed_labels") or []
            halluc = metrics.get("hallucinated_labels") or []
            parts = []
            if missed:
                parts.append(f"누락된 병변: {missed}. impression에 명시적으로 언급해야 한다.")
            if halluc:
                parts.append(f"근거 없는 추가 병변(hallucination): {halluc}. 이미지에 없으면 만들지 말아야 한다.")
            if metrics.get("normal_collapse"):
                parts.append("abnormal GT인데 normal로 결론냈다(normal collapse). 보이는 abnormal finding을 기술하라.")
            if not parts:
                parts.append("prediction이 GT impression과 대체로 일치한다. 간결성과 임상 용어 일관성만 유지하라.")
            return " ".join(parts)

        system = "You are a meticulous radiology QA reviewer comparing a generated chest X-ray impression to the reference."
        user = (
            f"Reference impression:\n{target}\n\n"
            f"Generated impression:\n{prediction}\n\n"
            f"Quantitative signals: {metrics}\n\n"
            "List concrete clinical differences: missed findings, hallucinated findings, and any normal collapse. "
            "Then give actionable feedback for improving the generation prompt's STYLE_PROFILE only "
            "(do not propose changing the model). Be concise."
        )
        return self._chat(system, user)

    def rewrite_style_profile(self, current_style_profile: str, critiques: list[str], metric_summary: dict) -> str:
        if self.backend in {"placeholder", "dummy", "mock"}:
            additions = []
            if metric_summary.get("normal_collapse_rate", 0.0) > 0.0:
                additions.append(
                    "When abnormal findings are visible, explicitly name them and never default to a normal impression."
                )
            if metric_summary.get("hallucination_rate", 0.0) > 0.0:
                additions.append("Only report findings clearly supported by the image; avoid unsupported conclusions.")
            if metric_summary.get("omission_rate", 0.0) > 0.0:
                additions.append("Prefer recall for clinically important findings; do not omit visible abnormalities.")
            if not additions:
                additions.append("Keep impressions concise and use consistent radiology terminology.")
            # 중복 라인 없이 합친다.
            base = current_style_profile.rstrip()
            for line in additions:
                if line not in base:
                    base += "\n" + line
            return base + "\n"

        system = (
            "You optimize a STYLE_PROFILE prompt fragment for a radiology report generator. "
            "The vision model is FROZEN — you change only the prompt text, never the model. "
            "Goal: make the generated IMPRESSION text better (clearer, more complete, matching the "
            "dataset's reporting style) WITHOUT changing the model's lesion-detection behavior — the set "
            "of findings it reports must stay statistically equivalent to the baseline (don't push it to "
            "report more or fewer abnormalities)."
        )
        joined = "\n- ".join(critiques[:20])
        user = (
            f"Current STYLE_PROFILE:\n{current_style_profile}\n\n"
            f"Aggregated metric summary: {metric_summary}\n\n"
            f"Critic feedback:\n- {joined}\n\n"
            "Rewrite an improved STYLE_PROFILE that improves impression wording/structure/completeness and "
            "dataset style, while keeping lesion detection unchanged (no increase in missed findings, "
            "hallucinations, or normal collapse vs baseline). Output ONLY the new STYLE_PROFILE text."
        )
        return self._chat(system, user)


def build_critic(config: dict) -> QwenCritic:
    """critic config에서 critic을 만든다. 현재는 Qwen 계열만 지원."""
    name = str(config.get("model_name", "")).lower()
    path = str(config.get("model_name_or_path", "")).lower()
    if "qwen" in name or "qwen" in path or not name:
        return QwenCritic(config)
    raise ValueError(f"지원하지 않는 critic입니다: {name}")
