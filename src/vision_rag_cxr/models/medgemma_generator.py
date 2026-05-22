"""MedGemma generator adapter.

MedGemma는 이 프레임워크에서 report generation과 localization JSON 생성을 담당하는 VLM이다.
실험 코드는 BaseGenerator interface만 호출하므로, 나중에 Qwen-VL/LLaVA-Med 등으로 바꿀 때는
새 generator adapter를 추가하고 config의 ``model_name``만 교체하면 된다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from vision_rag_cxr.models.base import BaseGenerator
from vision_rag_cxr.prompting.parser import parse_json_output


class MedGemmaGenerator(BaseGenerator):
    """MedGemma 기반 report/localization generator.

    ``backend: placeholder``는 pipeline 배선 확인용이다. 실제 연구 결과를 만들 때는
    ``backend: transformers``와 캐시된 MedGemma weight를 사용한다.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.backend = str(config.get("backend", "placeholder")).lower()
        self.model_name_or_path = config.get("model_name_or_path", "google/medgemma-4b-it")
        self._loaded = False
        self.model = None
        self.processor = None
        self.torch = None
        self.torch_dtype = None

    def load(self) -> None:
        """실제 모델을 GPU에 lazy loading한다."""
        if self._loaded:
            return
        if self.backend in {"placeholder", "dummy", "mock"}:
            self._loaded = True
            return
        if self.backend != "transformers":
            raise ValueError(f"지원하지 않는 MedGemma backend입니다: {self.backend}")

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        dtype_name = str(self.config.get("dtype", "bfloat16"))
        self.torch_dtype = getattr(torch, dtype_name, torch.bfloat16)
        local_files_only = bool(self.config.get("local_files_only", False))

        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=bool(self.config.get("trust_remote_code", False)),
        )

        model_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": bool(self.config.get("trust_remote_code", False)),
        }
        if self.config.get("device_map", "auto"):
            model_kwargs["device_map"] = self.config.get("device_map", "auto")

        max_memory = self.config.get("max_memory")
        if isinstance(max_memory, dict) and max_memory:
            # CUDA_VISIBLE_DEVICES를 쓰면 여기의 0,1,2...는 visible GPU의 local id다.
            model_kwargs["max_memory"] = {int(k) if str(k).isdigit() else k: v for k, v in max_memory.items()}

        quantization = str(self.config.get("quantization", "") or "").lower()
        load_in_4bit = bool(self.config.get("load_in_4bit", False)) or quantization in {"4bit", "bnb_4bit"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_quant_type=str(self.config.get("bnb_4bit_quant_type", "nf4")),
            )
        else:
            model_kwargs["torch_dtype"] = self.torch_dtype

        attn_impl = self.config.get("attn_implementation")
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        self.model = AutoModelForImageTextToText.from_pretrained(self.model_name_or_path, **model_kwargs)
        self.model.eval()
        self._loaded = True

    @staticmethod
    def _open_rgb(path: str | None) -> Image.Image | None:
        """이미지를 RGB PIL 객체로 연다. 없는 lateral은 조용히 skip한다."""
        if not path:
            return None
        p = Path(str(path))
        if not p.exists():
            return None
        return Image.open(p).convert("RGB")

    def _placeholder_impression(self) -> str:
        return json.dumps(
            {
                "impression": "Placeholder impression. Set backend: transformers for production experiment.",
                "mentioned_findings": [],
                "uncertainty_phrases": [],
                "no_finding_claim": False,
            },
            ensure_ascii=False,
        )

    def _placeholder_localization(self) -> dict:
        return {
            "lesions": [],
            "global_impression_optional": "Placeholder localization. Set backend: transformers for production experiment.",
        }

    def _context_image_content(self, context_examples: list[dict] | None) -> list[dict]:
        """RAG support example 이미지를 chat content에 추가한다.

        모든 support image를 넣으면 VRAM/latency가 급격히 커진다. 그래서 config의
        ``max_context_images``와 ``context_image_policy``로 어디까지 넣을지 제어한다.
        """
        if not self.config.get("include_context_images", False) or not context_examples:
            return []

        max_images = int(self.config.get("max_context_images", 2))
        policy = str(self.config.get("context_image_policy", "frontal")).lower()
        content: list[dict] = []
        used = 0
        for i, ex in enumerate(context_examples, start=1):
            if used >= max_images:
                break
            content.append({"type": "text", "text": f"Support example {i} image(s):"})
            frontal = self._open_rgb(ex.get("frontal_path"))
            if frontal is not None and used < max_images:
                content.append({"type": "image", "image": frontal})
                used += 1
            lateral = self._open_rgb(ex.get("lateral_path")) if policy in {"two_view", "frontal_lateral", "both"} else None
            if lateral is not None and used < max_images:
                content.append({"type": "image", "image": lateral})
                used += 1
        return content

    def _query_image_content(self, sample: dict) -> list[dict]:
        """query frontal/lateral 이미지를 chat content에 추가한다."""
        content: list[dict] = []
        frontal = self._open_rgb(sample.get("frontal_path"))
        lateral = self._open_rgb(sample.get("lateral_path"))
        if frontal is None:
            raise FileNotFoundError(f"frontal image not found for uid={sample.get('uid')}: {sample.get('frontal_path')}")
        content.extend([{"type": "text", "text": "Query frontal image:"}, {"type": "image", "image": frontal}])
        if lateral is not None:
            content.extend([{"type": "text", "text": "Query lateral image:"}, {"type": "image", "image": lateral}])
        return content

    def _generate_text(self, sample: dict, prompt: str, context_examples: list[dict] | None = None) -> str:
        """MedGemma chat template으로 text를 생성한다."""
        self.load()
        if self.backend in {"placeholder", "dummy", "mock"}:
            return ""

        system_prompt = self.config.get("system_prompt", "You are an expert radiologist.")
        content: list[dict] = [{"type": "text", "text": prompt}]
        content.extend(self._context_image_content(context_examples))
        content.extend(self._query_image_content(sample))

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device, dtype=self.torch_dtype)
        input_len = inputs["input_ids"].shape[-1]

        temperature = float(self.config.get("temperature", 0.0))
        generation_kwargs = {
            "max_new_tokens": int(self.config.get("max_new_tokens", 512)),
            "do_sample": temperature > 0.0,
            "temperature": temperature if temperature > 0.0 else None,
            "top_p": float(self.config.get("top_p", 1.0)),
        }
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

        with self.torch.inference_mode():
            generation = self.model.generate(**inputs, **generation_kwargs)
        new_tokens = generation[0][input_len:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_impression(self, sample: dict, prompt: str, context_examples: list[dict] | None = None) -> str:
        """impression JSON 문자열을 생성한다."""
        self.load()
        if self.backend in {"placeholder", "dummy", "mock"}:
            return self._placeholder_impression()
        return self._generate_text(sample, prompt, context_examples=context_examples)

    def generate_localization(self, sample: dict, prompt: str, context_examples: list[dict] | None = None) -> dict:
        """localization JSON을 생성하고 가능하면 dict로 parsing한다."""
        self.load()
        if self.backend in {"placeholder", "dummy", "mock"}:
            return self._placeholder_localization()

        raw = self._generate_text(sample, prompt, context_examples=context_examples)
        parsed, err = parse_json_output(raw)
        if parsed is not None:
            return parsed
        return {"lesions": [], "raw_text": raw, "parse_error": err or "failed_to_parse_json"}
