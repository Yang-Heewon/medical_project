"""Generic HuggingFace VLM generator.

MedGemma/Qwen2.5-VL/LLaVA-Med 등 ``AutoModelForImageTextToText`` + ``AutoProcessor``
chat-template 계열 VLM을 공통으로 다루는 어댑터다. 모델별로 다른 건 default model id와
system prompt 정도라서, 이 base 하나로 plug-and-play 교체가 된다.

backend:
- ``placeholder``(기본): 모델 없이 pipeline/배선 검증. JSON placeholder 반환.
- ``transformers``: 실제 weight 로드 후 image+prompt로 생성.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from vision_rag_cxr.models.base import BaseGenerator
from vision_rag_cxr.prompting.parser import parse_json_output


class HFVLMGenerator(BaseGenerator):
    """AutoModelForImageTextToText 계열 공통 VLM 어댑터."""

    default_model = "google/medgemma-4b-it"
    default_system_prompt = "You are an expert radiologist."

    def __init__(self, config: dict):
        super().__init__(config)
        self.backend = str(config.get("backend", "placeholder")).lower()
        self.model_name_or_path = config.get("model_name_or_path", self.default_model)
        self._loaded = False
        self.model = None
        self.processor = None
        self.torch = None
        self.torch_dtype = None

    def load(self) -> None:
        if self._loaded:
            return
        if self.backend in {"placeholder", "dummy", "mock"}:
            self._loaded = True
            return
        if self.backend != "transformers":
            raise ValueError(f"지원하지 않는 VLM backend입니다: {self.backend}")

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from vision_rag_cxr.utils.devices import resolve_device, resolve_dtype

        self.torch = torch
        self.device = resolve_device(self.config.get("device", "auto"))
        self.torch_dtype = resolve_dtype(self.device, self.config.get("dtype"))
        local_files_only = bool(self.config.get("local_files_only", False))
        trust = bool(self.config.get("trust_remote_code", False))

        # 이미지 사이즈는 VLM 자기 프로세서의 네이티브 리사이즈에 맡긴다(공정성: 같은 VLM이 받는 사이즈로).
        # Qwen2.5-VL류는 min_pixels/max_pixels(28-grid 픽셀 예산)로 동적 리사이즈한다.
        proc_kwargs: dict[str, Any] = {"local_files_only": local_files_only, "trust_remote_code": trust}
        for k in ("min_pixels", "max_pixels"):
            if self.config.get(k) is not None:
                proc_kwargs[k] = int(self.config[k])
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path, **proc_kwargs)
        model_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only, "trust_remote_code": trust, "torch_dtype": self.torch_dtype,
        }
        attn_impl = self.config.get("attn_implementation")
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        quantization = str(self.config.get("quantization", "") or "").lower()
        want_4bit = bool(self.config.get("load_in_4bit", False)) or quantization in {"4bit", "bnb_4bit"}

        if self.device == "cuda":
            # CUDA: accelerate device_map + (선택) bitsandbytes 4bit
            if self.config.get("device_map", "auto"):
                model_kwargs["device_map"] = self.config.get("device_map", "auto")
            max_memory = self.config.get("max_memory")
            if isinstance(max_memory, dict) and max_memory:
                model_kwargs["max_memory"] = {int(k) if str(k).isdigit() else k: v for k, v in max_memory.items()}
            if want_4bit:
                from transformers import BitsAndBytesConfig
                model_kwargs.pop("torch_dtype", None)
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=self.torch_dtype,
                    bnb_4bit_quant_type=str(self.config.get("bnb_4bit_quant_type", "nf4")),
                )
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_name_or_path, **model_kwargs)
        else:
            # MPS(Apple)/CPU: device_map·bitsandbytes 미사용, 로드 후 .to(device)
            if want_4bit:
                print(f"[hf_vlm] {self.device}에서는 4bit(bitsandbytes) 불가 -> fp 풀로드로 진행", flush=True)
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_name_or_path, **model_kwargs)
            self.model = self.model.to(self.device)

        self.model.eval()
        self._loaded = True

    def _open_rgb(self, path: str | None) -> Image.Image | None:
        if not path:
            return None
        p = Path(str(path))
        if not p.exists():
            return None
        img = Image.open(p).convert("RGB")
        # 기본은 PIL 선축소 없이 원본을 넘겨, VLM 프로세서가 자기 네이티브 사이즈로 리사이즈하게 한다
        # (공정성: 같은 VLM이 실제로 받는 해상도). 토큰/메모리 상한은 processor의 max_pixels로 묶는다.
        # max_image_size를 명시적으로 0이 아닌 값으로 주면 (레거시) 긴 변을 그 값으로 선축소한다.
        max_side = int(self.config.get("max_image_size", 0) or 0)
        if max_side and max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.BILINEAR)
        return img

    def _placeholder_impression(self) -> str:
        return json.dumps(
            {
                "impression": f"Placeholder impression ({self.model_name}). Set backend: transformers for production.",
                "mentioned_findings": [],
                "uncertainty_phrases": [],
                "no_finding_claim": False,
            },
            ensure_ascii=False,
        )

    def _placeholder_localization(self) -> dict:
        return {"lesions": [], "global_impression_optional": f"Placeholder localization ({self.model_name})."}

    def _context_image_content(self, context_examples: list[dict] | None) -> list[dict]:
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
        self.load()
        if self.backend in {"placeholder", "dummy", "mock"}:
            return ""
        system_prompt = self.config.get("system_prompt", self.default_system_prompt)
        content: list[dict] = [{"type": "text", "text": prompt}]
        content.extend(self._context_image_content(context_examples))
        content.extend(self._query_image_content(sample))
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
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
        self.load()
        if self.backend in {"placeholder", "dummy", "mock"}:
            return self._placeholder_impression()
        return self._generate_text(sample, prompt, context_examples=context_examples)

    def generate_localization(self, sample: dict, prompt: str, context_examples: list[dict] | None = None) -> dict:
        self.load()
        if self.backend in {"placeholder", "dummy", "mock"}:
            return self._placeholder_localization()
        raw = self._generate_text(sample, prompt, context_examples=context_examples)
        parsed, err = parse_json_output(raw)
        if parsed is not None:
            return parsed
        return {"lesions": [], "raw_text": raw, "parse_error": err or "failed_to_parse_json"}


class Qwen25VLGenerator(HFVLMGenerator):
    default_model = "Qwen/Qwen2.5-VL-7B-Instruct"
    default_system_prompt = "You are an expert radiologist analyzing chest X-rays."


class LlavaMedGenerator(HFVLMGenerator):
    default_model = "microsoft/llava-med-v1.5-mistral-7b"
    default_system_prompt = "You are an expert radiologist analyzing chest X-rays."
