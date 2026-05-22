"""Final RAG vs No-RAG experiment."""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from vision_rag_cxr.experiments.experiment_base import ExperimentBase
from vision_rag_cxr.models.vlm_generator import build_generator
from vision_rag_cxr.prompting.parser import parse_json_output
from vision_rag_cxr.prompting.prompt_templates import BASE_STYLE_PROFILE, IMPRESSION_PROMPT
from vision_rag_cxr.rag.prompt_context_builder import build_context_examples_text
from vision_rag_cxr.rag.retriever_factory import build_retriever_config
from vision_rag_cxr.rag.related_retriever import RelatedRetriever


def _style_profile(config: dict) -> str:
    path = config.get("optimized_style_profile_path")
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8")
    return config.get("style_profile", BASE_STYLE_PROFILE)


class FinalRAGExperiment(ExperimentBase):
    """No-RAG와 Vision-RAG를 같은 inference set에서 paired 비교한다."""

    def run(self, generator_config: dict):
        generator = build_generator(generator_config)
        df = self.load_inference_set()
        style_profile = _style_profile(self.config)

        retriever_cfg = build_retriever_config({**self.config, "rag_mode": "related"})
        retriever = RelatedRetriever.from_config(retriever_cfg)
        rows = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=self.config["experiment_name"]):
            sample = row.to_dict()

            for condition, context_examples in [
                ("no_rag", []),
                ("vision_rag", retriever.retrieve(sample, self.config.get("top_k", 5))),
            ]:
                context_text = build_context_examples_text(context_examples)
                prompt = IMPRESSION_PROMPT.format(style_profile=style_profile, context_examples=context_text)
                raw = generator.generate_impression(sample, prompt, context_examples=context_examples)
                parsed, err = parse_json_output(raw)
                rows.append(
                    {
                        "uid": row["uid"],
                        "experiment_name": self.config["experiment_name"],
                        "condition": condition,
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
