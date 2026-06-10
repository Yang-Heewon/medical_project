"""Impression generation experiment."""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from vision_rag_cxr.inference.experiments.experiment_base import ExperimentBase
from vision_rag_cxr.models.generators.factory import build_generator
from vision_rag_cxr.prompting.parser import parse_json_output
from vision_rag_cxr.prompting.prompt_templates import BASE_STYLE_PROFILE, IMPRESSION_PROMPT, render_prompt
from vision_rag_cxr.inference.retrieval.prompt_context_builder import build_context_examples_text
from vision_rag_cxr.inference.retrieval.retriever_factory import build_retriever_for_experiment


def _style_profile_from_config(config: dict) -> str:
    """TextGrad가 저장한 style profile이 있으면 사용하고, 없으면 기본값을 쓴다."""
    path = config.get("optimized_style_profile_path")
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8")
    return config.get("style_profile", BASE_STYLE_PROFILE)


class ImpressionExperiment(ExperimentBase):
    def run(self, generator_config: dict):
        generator = build_generator(generator_config)
        df = self.load_inference_set()
        retriever = build_retriever_for_experiment(self.config)
        style_profile = _style_profile_from_config(self.config)
        rows = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=self.config["experiment_name"]):
            sample = row.to_dict()
            context_examples = retriever.retrieve(sample, self.config.get("top_k", 5)) if retriever else []
            context_text = build_context_examples_text(context_examples)
            prompt = render_prompt(IMPRESSION_PROMPT, style_profile=style_profile,
                                   context_examples=context_text, modality=sample.get("modality"))
            raw = generator.generate_impression(sample, prompt, context_examples=context_examples)
            parsed, err = parse_json_output(raw)

            rows.append(
                {
                    "uid": row["uid"],
                    "experiment_name": self.config["experiment_name"],
                    "rag_mode": self.config.get("rag_mode", "image_only"),
                    "prompt_version": self.config.get("prompt_version", "v1"),
                    "model_name": generator.model_name,
                    "retrieved_uids": json.dumps([ex.get("uid") for ex in context_examples], ensure_ascii=False),
                    "retrieval_scores": json.dumps([ex.get("retrieval_score") for ex in context_examples], ensure_ascii=False),
                    "raw_output": raw,
                    "parsed_output": json.dumps(parsed, ensure_ascii=False) if parsed else "",
                    "parse_error": err or "",
                    "gt_impression": row.get("impression", ""),
                    "gt_labels": row.get("chexbert_labels_binary", ""),
                }
            )

        self.save_predictions(rows)
