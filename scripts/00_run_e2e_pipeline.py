#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Vision-RAG-CXR 전체 E2E pipeline orchestrator.

이 스크립트가 프로젝트의 단일 진입점이다.

Smoke mode:
- 전처리, split, RAG DB, TextGrad placeholder, 6개 ablation, final paired run을 모두 실행한다.
- `max_inference_samples`로 inference 일부만 사용해 배선 오류를 빠르게 잡는다.

Full mode:
- 같은 단계를 전체 inference set에 적용한다.
- 실제 논문/보고 결과에는 MedGemma generation, CheXbert label merge, production retrieval encoder가 연결되어 있어야 한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from vision_rag_cxr.data.indiana_preprocessor import preprocess_indiana
from vision_rag_cxr.data.splitters import create_splits
from vision_rag_cxr.experiments.final_rag_experiment import FinalRAGExperiment
from vision_rag_cxr.experiments.impression_experiment import ImpressionExperiment
from vision_rag_cxr.experiments.localization_experiment import LocalizationExperiment
from vision_rag_cxr.experiments.prompt_optimization_experiment import run_prompt_optimization
from vision_rag_cxr.experiments.rag_ablation_experiment import RAGAblationExperiment
from vision_rag_cxr.rag.build_database import build_support_database
from vision_rag_cxr.utils.io import ensure_dir, load_yaml


EXPERIMENTS = [
    {
        "experiment_name": "image_only_localization",
        "task_type": "localization",
        "rag_mode": "image_only",
        "runner": "baseline",
    },
    {
        "experiment_name": "image_only_impression",
        "task_type": "impression",
        "rag_mode": "image_only",
        "runner": "baseline",
    },
    {
        "experiment_name": "unrelated_rag_localization",
        "task_type": "localization",
        "rag_mode": "unrelated",
        "runner": "rag",
    },
    {
        "experiment_name": "unrelated_rag_impression",
        "task_type": "impression",
        "rag_mode": "unrelated",
        "runner": "rag",
    },
    {
        "experiment_name": "related_rag_localization",
        "task_type": "localization",
        "rag_mode": "related",
        "runner": "rag",
    },
    {
        "experiment_name": "related_rag_impression",
        "task_type": "impression",
        "rag_mode": "related",
        "runner": "rag",
    },
]


def _stage(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def _path(work_dir: Path, *parts: str) -> Path:
    return work_dir.joinpath(*parts)


def _load(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def _common_experiment_config(
    pipe: dict[str, Any],
    work_dir: Path,
    split_csv: Path,
    rag_dir: Path,
    experiment_name: str,
    task_type: str,
    rag_mode: str,
) -> dict[str, Any]:
    """각 experiment가 공유하는 경로/config override를 만든다."""
    max_samples = pipe.get("max_inference_samples")
    cfg = {
        "experiment_name": experiment_name,
        "task_type": task_type,
        "rag_mode": rag_mode,
        "generator_config": pipe["configs"]["generator"],
        "critic_config": pipe["configs"].get("critic"),
        "retrieval_config": pipe["configs"]["retrieval"],
        "split_csv": str(split_csv),
        "support_metadata_path": str(rag_dir / "support_metadata.parquet"),
        "image_embeddings_path": str(rag_dir / "image_embeddings.npy"),
        "text_embeddings_path": str(rag_dir / "text_embeddings.npy"),
        "label_vectors_path": str(rag_dir / "label_vectors.npy"),
        "top_k": int(pipe.get("top_k", 5)),
        "seed": int(pipe.get("seed", 0)),
        "prompt_version": "pipeline_v1",
        "output_dir": str(_path(work_dir, "experiments", experiment_name)),
    }
    if max_samples is not None:
        cfg["max_inference_samples"] = int(max_samples)

    optimized = _path(work_dir, "prompt_optimization", "optimized_style_profile.txt")
    if optimized.exists():
        cfg["optimized_style_profile_path"] = str(optimized)
    return cfg


def _run_one_experiment(cfg: dict[str, Any], generator_cfg: dict[str, Any]) -> None:
    if cfg.get("rag_mode") in {"unrelated", "related"}:
        RAGAblationExperiment(cfg).run(generator_cfg)
        return
    if cfg["task_type"] == "localization":
        LocalizationExperiment(cfg).run(generator_cfg)
        return
    if cfg["task_type"] == "impression":
        ImpressionExperiment(cfg).run(generator_cfg)
        return
    raise ValueError(f"Unknown task_type: {cfg['task_type']}")


def _summarize(work_dir: Path) -> None:
    """각 predictions.csv를 하나로 합쳐 smoke/final 산출물 위치를 명확히 남긴다."""
    summary_dir = ensure_dir(_path(work_dir, "summary"))
    files = sorted(_path(work_dir, "experiments").glob("**/predictions.csv"))
    final_file = _path(work_dir, "final_rag_vs_no_rag", "predictions.csv")
    if final_file.exists():
        files.append(final_file)

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        frames.append(df)

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
    else:
        all_df = pd.DataFrame()
    all_df.to_csv(summary_dir / "main_results.csv", index=False)

    report = [
        "# E2E Pipeline Summary",
        "",
        f"- prediction_files: {len(files)}",
        f"- total_prediction_rows: {len(all_df)}",
        "",
        "## Files",
    ]
    report.extend([f"- `{path}`" for path in files])
    report.extend(
        [
            "",
            "## Production readiness note",
            "Smoke success means the pipeline wiring works. It does not mean the scientific result is valid until production model adapters are connected.",
        ]
    )
    (summary_dir / "final_report.md").write_text("\n".join(report), encoding="utf-8")


def run_pipeline(config_path: str | Path) -> dict[str, str]:
    pipe = _load(config_path)
    work_dir = ensure_dir(pipe.get("work_dir", "outputs/e2e_smoke"))
    run_flags = pipe.get("run", {})

    pre_dir = ensure_dir(_path(work_dir, "preprocessed"))
    split_dir = ensure_dir(_path(work_dir, "splits"))
    rag_dir = ensure_dir(_path(work_dir, "rag"))
    prompt_dir = ensure_dir(_path(work_dir, "prompt_optimization"))
    ensure_dir(_path(work_dir, "experiments"))

    _stage(f"Pipeline start: {pipe.get('pipeline_name', config_path)}")
    print(f"work_dir: {work_dir}", flush=True)

    data_csv_override = pipe.get("data_csv_override") or pipe.get("input_data_csv")
    data_csv = Path(data_csv_override) if data_csv_override else pre_dir / "indiana_paired_samples.csv"
    if data_csv_override:
        _stage("1/7 use externally prepared paired samples")
        if not data_csv.exists():
            raise FileNotFoundError(f"data_csv_override not found: {data_csv}")
        print(f"data_csv_override: {data_csv}", flush=True)
    elif run_flags.get("preprocess", True):
        _stage("1/7 preprocess Indiana paired samples")
        data_cfg = _load(pipe["configs"]["data"])
        data_cfg["output_dir"] = str(pre_dir)
        preprocess_indiana(data_cfg)

    split_csv = split_dir / f"split_seed_{int(pipe.get('seed', 0))}.csv"
    if run_flags.get("split", True):
        _stage("2/7 create support/inference split")
        split_cfg = _load(pipe["configs"]["split"])
        split_cfg["output_dir"] = str(split_dir)
        split_cfg["seeds"] = pipe.get("seeds", [pipe.get("seed", 0)])
        create_splits(str(data_csv), split_cfg)

    if run_flags.get("build_rag_db", True):
        _stage("3/7 build support RAG database")
        retrieval_cfg = _load(pipe["configs"]["retrieval"])
        retrieval_cfg["output_dir"] = str(rag_dir)
        if pipe.get("max_support_samples") is not None:
            retrieval_cfg["max_support_samples"] = int(pipe["max_support_samples"])
        build_support_database(str(split_csv), retrieval_cfg)

    if run_flags.get("textgrad", True):
        _stage("4/7 run TextGrad prompt optimization placeholder")
        textgrad_cfg = _load(pipe["configs"]["textgrad"])
        textgrad_cfg["split_csv"] = str(split_csv)
        textgrad_cfg["output_dir"] = str(prompt_dir)
        textgrad_cfg["generator_config"] = pipe["configs"]["generator"]
        textgrad_cfg["critic_config"] = pipe["configs"].get("critic")
        run_prompt_optimization(textgrad_cfg)

    generator_cfg = _load(pipe["configs"]["generator"])
    if run_flags.get("six_experiments", True):
        _stage("5/7 run six baseline/RAG ablation experiments")
        for exp in EXPERIMENTS:
            exp_cfg = _common_experiment_config(
                pipe,
                work_dir,
                split_csv,
                rag_dir,
                exp["experiment_name"],
                exp["task_type"],
                exp["rag_mode"],
            )
            print(f"running: {exp_cfg['experiment_name']} ({exp_cfg['rag_mode']})", flush=True)
            _run_one_experiment(exp_cfg, generator_cfg)

    if run_flags.get("final_rag_vs_no_rag", True):
        _stage("6/7 run final paired No-RAG vs Vision-RAG")
        final_cfg = _common_experiment_config(
            pipe,
            work_dir,
            split_csv,
            rag_dir,
            "final_rag_vs_no_rag",
            "impression",
            "related",
        )
        final_cfg["output_dir"] = str(_path(work_dir, "final_rag_vs_no_rag"))
        FinalRAGExperiment(final_cfg).run(generator_cfg)

    if run_flags.get("summarize", True):
        _stage("7/7 summarize outputs")
        _summarize(work_dir)

    manifest = {
        "work_dir": str(work_dir),
        "preprocessed_csv": str(data_csv),
        "split_csv": str(split_csv),
        "rag_dir": str(rag_dir),
        "experiments_dir": str(_path(work_dir, "experiments")),
        "final_dir": str(_path(work_dir, "final_rag_vs_no_rag")),
        "summary_dir": str(_path(work_dir, "summary")),
    }
    (work_dir / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _stage("Pipeline completed")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline/e2e_smoke.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
