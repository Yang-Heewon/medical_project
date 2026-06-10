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
        # critic이 인지하는 데이터셋 라벨 공간(이 데이터셋이 보고하는 finding 어휘).
        self.label_space = str(config.get("label_space", "")) or "(unspecified)"
        self.label_names = list(config.get("label_names", []) or [])
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

    def _label_awareness(self) -> str:
        """critic 프롬프트에 주입할 '데이터셋 label space 인지' 문구.

        이 데이터셋이 실제로 보고하는 finding 어휘를 알려, STYLE_PROFILE이 그 데이터셋의
        보고 카테고리/용어에 맞춰 최적화되도록 한다 (병변 검출 자체는 바꾸지 않는다)."""
        if not self.label_names:
            return ""
        shown = ", ".join(self.label_names[:40])
        return (
            f"\nThis dataset's report label space is '{self.label_space}'. The findings it actually "
            f"reports are: {shown}. Tailor wording/terminology to how THESE findings are phrased in this "
            f"dataset, and do not introduce finding categories outside this space. The frozen model's "
            f"lesion detection over these categories must stay unchanged.\n"
        )

    # --- critic API ----------------------------------------------------------
    def critique(self, prediction: str, target: str, sample: dict, metrics: dict) -> str:
        # plug-in/out: 무엇을 비평할지 (config로 교체)
        #   "style"   = 생성 impression vs 참조 impression(target=GT)의 표현/구조/길이 차이 (기본)
        #   "clinical"= 누락/환각/normal collapse 등 임상 소견 차이
        focus = str(self.config.get("critique_focus", "style")).lower()

        if self.backend in {"placeholder", "dummy", "mock"}:
            if focus == "clinical":
                missed = metrics.get("missed_labels") or []
                halluc = metrics.get("hallucinated_labels") or []
                parts = []
                if missed:
                    parts.append(f"누락된 병변: {missed}. impression에 명시적으로 언급해야 한다.")
                if halluc:
                    parts.append(f"근거 없는 추가 병변(hallucination): {halluc}. 이미지에 없으면 만들지 말아야 한다.")
                if metrics.get("normal_collapse"):
                    parts.append("abnormal GT인데 normal로 결론냈다(normal collapse). 보이는 abnormal finding을 기술하라.")
                return " ".join(parts) or "prediction이 GT와 대체로 일치. 간결성/용어 일관성 유지."
            # style
            gp, pp = str(target or ""), str(prediction or "")
            note = []
            if len(pp) > len(gp) * 1.3:
                note.append("생성 impression이 참조보다 장황하다. 참조처럼 간결하게.")
            elif len(pp) < len(gp) * 0.7:
                note.append("생성 impression이 참조보다 너무 짧다. 참조 수준의 정보량으로.")
            note.append("참조 impression의 문장 구조·용어·어조에 맞춰 표현하라(병변 검출은 그대로 유지).")
            return " ".join(note)

        if focus == "clinical":
            system = "You are a meticulous radiology QA reviewer comparing a generated chest X-ray impression to the reference."
            user = (
                f"Reference impression:\n{target}\n\n"
                f"Generated impression:\n{prediction}\n\n"
                f"Quantitative signals: {metrics}\n\n"
                "List concrete clinical differences: missed findings, hallucinated findings, normal collapse. "
                "Then actionable feedback for the STYLE_PROFILE only. Be concise."
            )
            return self._chat(system, user)

        # focus == "style" (기본)
        system = (
            "You are a radiology report editor. You compare a generated chest X-ray IMPRESSION to the gold "
            "reference impression and improve how it is WORDED — not which findings the model detects."
        )
        user = (
            f"Reference impression (dataset gold style):\n{target}\n\n"
            f"Generated impression:\n{prediction}\n\n"
            f"{self._label_awareness()}"
            "Point out concrete STYLE differences vs the reference: wording, structure/sectioning, length/conciseness, "
            "terminology, phrasing, punctuation, formatting. Give actionable feedback to make the generated impression's "
            "STYLE match the reference. Do NOT ask to add or remove clinical findings — the model's lesion detection "
            "must stay unchanged. Be concise."
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
            f"{self._label_awareness()}"
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
