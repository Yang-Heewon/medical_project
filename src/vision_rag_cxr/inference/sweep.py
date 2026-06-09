#!/usr/bin/env python
"""실험 sweep 러너 — {seed × encoder × generator × rag_mode × task} 매트릭스를 무인 실행.

특징:
- 순차 실행(GPU 1 결함 회피용으로 CUDA_VISIBLE_DEVICES는 호출측에서 0 등으로 고정).
- 각 조합 실패해도 다음으로 계속(전체 sweep 안 죽음). 에러는 results.csv에 기록.
- resumable: results.csv에 status=ok로 끝난 조합은 skip.
- split/RAG DB는 (seed, encoder)별로 한 번만 만들고 캐시.

진행 상황: <out_dir>/results.csv, <out_dir>/sweep_state.json, 조합별 로그 <out_dir>/logs/.
"""
from __future__ import annotations
import argparse, json, sys, traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from vision_rag_cxr.datasets.splitters import create_splits
from vision_rag_cxr.datasets.labeler_chexbert import CheXbertLikeLabeler
from vision_rag_cxr.datasets.label_spaces import resolve_labels
from vision_rag_cxr.evaluation.chexbert_metrics import labels_json_to_matrix, multilabel_scores
from vision_rag_cxr.inference.experiments.impression_experiment import ImpressionExperiment
from vision_rag_cxr.inference.experiments.localization_experiment import LocalizationExperiment
from vision_rag_cxr.inference.experiments.rag_ablation_experiment import RAGAblationExperiment
from vision_rag_cxr.inference.retrieval.build_database import build_support_database
from vision_rag_cxr.utils.io import ensure_dir, load_yaml


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(out_dir, msg):
    print(f"[{_now()}] {msg}", flush=True)
    with open(Path(out_dir) / "sweep.log", "a", encoding="utf-8") as f:
        f.write(f"[{_now()}] {msg}\n")


def _impression_metrics(pred_csv: str, labels: list[str]) -> dict:
    """생성 impression에서 예측 label을 뽑아 GT label과 micro/macro F1 계산."""
    df = pd.read_csv(pred_csv)
    lab = CheXbertLikeLabeler()
    preds, gts = [], []
    for _, r in df.iterrows():
        raw = str(r.get("parsed_output") or r.get("raw_output") or "")
        try:
            obj = json.loads(raw)
            text = str(obj.get("impression", "")) + " " + " ".join(map(str, obj.get("mentioned_findings", [])))
        except Exception:
            text = raw
        b, _ = lab.label_report(text)
        preds.append(json.dumps(b))
        gts.append(str(r.get("gt_labels") or "{}"))
    if not preds:
        return {}
    y_pred = labels_json_to_matrix(preds, labels)
    y_true = labels_json_to_matrix(gts, labels)
    return multilabel_scores(y_true, y_pred)


def run_sweep(cfg: dict):
    out_dir = ensure_dir(cfg["out_dir"])
    ensure_dir(Path(out_dir) / "logs")
    data_csv = cfg["data_csv"]
    label_space = cfg.get("label_space", "chexbert_14")
    labels = resolve_labels({"label_space": label_space})
    seeds = cfg.get("seeds", [0])
    encoders = cfg["encoders"]          # list of retrieval configs (dict, vision_encoder_name 포함)
    generators = cfg["generators"]      # list of generator configs (dict, backend 포함)
    rag_modes = cfg.get("rag_modes", ["image_only", "unrelated", "related"])
    tasks = cfg.get("tasks", ["impression", "localization"])
    top_k = int(cfg.get("top_k", 5))
    max_inf = cfg.get("max_inference_samples")
    shard_id = int(cfg.get("_shard_id", 0))
    num_shards = int(cfg.get("_num_shards", 1))
    build_only = bool(cfg.get("_build_only", False))

    # 멀티-GPU: 각 worker(shard)가 자기 results 파일에 기록해 동시 append race를 피한다.
    results_path = Path(out_dir) / (f"results_shard{shard_id}.csv" if num_shards > 1 else "results.csv")
    done = set()
    if results_path.exists():
        prev = pd.read_csv(results_path)
        done = set(prev[prev["status"] == "ok"]["combo_id"].astype(str))
        _log(out_dir, f"[shard {shard_id}/{num_shards}] resume: {len(done)} combos already ok")

    def append_result(row: dict):
        row["timestamp"] = _now()
        hdr = not results_path.exists()
        pd.DataFrame([row]).to_csv(results_path, mode="a", header=hdr, index=False)

    # 1) seed별 split + encoder별 DB (캐시)
    split_csvs, rag_dirs = {}, {}
    for seed in seeds:
        sdir = ensure_dir(Path(out_dir) / f"seed_{seed}" / "splits")
        scsv = sdir / f"split_seed_{seed}.csv"
        if not scsv.exists():
            _log(out_dir, f"split seed={seed}")
            create_splits(data_csv, {"support_ratio": cfg.get("support_ratio", 0.7), "seeds": [seed],
                                     "output_dir": str(sdir), "label_space": label_space})
        split_csvs[seed] = str(scsv)
        for enc in encoders:
            tag = enc.get("benchmark_tag", enc["vision_encoder_name"])
            rdir = ensure_dir(Path(out_dir) / f"seed_{seed}" / f"rag_{tag}")
            rag_dirs[(seed, tag)] = rdir
            if (rdir / "image_embeddings.npy").exists():
                continue
            try:
                _log(out_dir, f"build DB seed={seed} encoder={tag}")
                ec = dict(enc); ec["output_dir"] = str(rdir); ec["label_space"] = label_space
                ec["query_label_source"] = "none"; ec["query_text_source"] = "none"
                build_support_database(split_csvs[seed], ec)
            except Exception as e:
                _log(out_dir, f"DB build FAILED seed={seed} encoder={tag}: {e}")

    if build_only:
        (Path(out_dir) / "BUILD_READY").write_text(_now() + "\n", encoding="utf-8")
        _log(out_dir, "build_only done (split+DB ready)")
        return

    # 2) 실험 매트릭스를 평탄화한 뒤 shard로 분배(GPU별 worker가 idx%num_shards==shard_id만 실행).
    combos = []
    for seed in seeds:
        for gen in generators:
            gen_tag = str(gen.get("model_name", gen.get("model_name_or_path", "gen"))).replace("/", "_")
            for rag_mode in rag_modes:
                enc_list = [{"benchmark_tag": "none"}] if rag_mode == "image_only" else encoders
                for enc in enc_list:
                    enc_tag = enc.get("benchmark_tag", enc.get("vision_encoder_name", "none"))
                    for task in tasks:
                        combos.append((seed, gen, gen_tag, rag_mode, enc, enc_tag, task))

    for idx, (seed, gen, gen_tag, rag_mode, enc, enc_tag, task) in enumerate(combos):
        if idx % num_shards != shard_id:
            continue
        combo_id = f"s{seed}|{gen_tag}|{enc_tag}|{rag_mode}|{task}"
        if combo_id in done:
            continue
        exp_name = f"{gen_tag}__{enc_tag}__{rag_mode}__{task}__seed{seed}"
        exp_dir = ensure_dir(Path(out_dir) / "experiments" / exp_name)
        rdir = rag_dirs.get((seed, enc_tag))
        cfg_exp = {
            "experiment_name": exp_name, "task_type": task, "rag_mode": rag_mode,
            "generator_config": gen, "retrieval_config": enc.get("_config_path", ""),
            "split_csv": split_csvs[seed], "top_k": top_k, "seed": seed,
            "prompt_version": "sweep_v1", "output_dir": str(exp_dir),
        }
        # TextGrad로 최적화된 STYLE_PROFILE이 있으면 No-RAG/RAG 생성에 사용 (textgrad-first 순서)
        if cfg.get("optimized_style_profile_path"):
            cfg_exp["optimized_style_profile_path"] = cfg["optimized_style_profile_path"]
        if rdir is not None:
            cfg_exp.update({
                "support_metadata_path": str(rdir / "support_metadata.parquet"),
                "image_embeddings_path": str(rdir / "image_embeddings.npy"),
                "text_embeddings_path": str(rdir / "text_embeddings.npy"),
                "label_vectors_path": str(rdir / "label_vectors.npy"),
            })
        if max_inf is not None:
            cfg_exp["max_inference_samples"] = int(max_inf)
        _log(out_dir, f"[shard {shard_id}] RUN {combo_id}")
        try:
            if rag_mode in {"unrelated", "related"}:
                RAGAblationExperiment(cfg_exp).run(gen)
            elif task == "localization":
                LocalizationExperiment(cfg_exp).run(gen)
            else:
                ImpressionExperiment(cfg_exp).run(gen)
            pred_csv = str(exp_dir / "predictions.csv")
            metrics = _impression_metrics(pred_csv, labels) if task == "impression" else {}
            row = {"combo_id": combo_id, "seed": seed, "generator": gen_tag,
                   "encoder": enc_tag, "rag_mode": rag_mode, "task": task,
                   "status": "ok", "predictions": pred_csv, "error": ""}
            row.update(metrics)
            append_result(row)
            _log(out_dir, f"OK  {combo_id}  {metrics}")
        except Exception as e:
            tb = traceback.format_exc()
            (Path(out_dir) / "logs" / f"{exp_name}.err").write_text(tb, encoding="utf-8")
            append_result({"combo_id": combo_id, "seed": seed, "generator": gen_tag,
                            "encoder": enc_tag, "rag_mode": rag_mode, "task": task,
                            "status": "failed", "predictions": "", "error": f"{type(e).__name__}: {e}"})
            _log(out_dir, f"FAIL {combo_id}: {type(e).__name__}: {e}")

    marker = f"SHARD_DONE_{shard_id}" if num_shards > 1 else "SWEEP_DONE"
    (Path(out_dir) / marker).write_text(_now() + "\n", encoding="utf-8")
    _log(out_dir, f"[shard {shard_id}/{num_shards}] COMPLETE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--build_only", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    cfg["_shard_id"] = args.shard_id
    cfg["_num_shards"] = args.num_shards
    cfg["_build_only"] = args.build_only
    # encoder retrieval config를 파일로 떨궈 experiment의 retrieval_config 경로로 쓴다.
    out_dir = ensure_dir(cfg["out_dir"])
    cfg_dir = ensure_dir(Path(out_dir) / "encoder_configs")
    import yaml
    for enc in cfg.get("encoders", []):
        tag = enc.get("benchmark_tag", enc["vision_encoder_name"])
        p = cfg_dir / f"{tag}.yaml"
        p.write_text(yaml.safe_dump(enc), encoding="utf-8")
        enc["_config_path"] = str(p)
    run_sweep(cfg)


if __name__ == "__main__":
    main()
