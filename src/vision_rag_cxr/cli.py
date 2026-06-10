"""Vision-RAG-CXR 통합 CLI — 3가지 옵션 구조.

  vrag list                        # 끼울 수 있는 dataset / generator / encoder 목록 (plug-in/out 카탈로그)
  vrag build  --dataset NAME ...   # ① 데이터 구축 (canonical CSV 생성)
  vrag infer  --dataset-csv CSV --generators ... --encoder ... --modes ...   # ② inference (No-RAG vs Vision-RAG)

핵심 설계: dataset / generator / encoder가 모두 레지스트리 기반이라 이름만 바꿔 끼우고 뺀다(plug-in/plug-out).
inference의 기본 비교는 No-RAG(image_only) vs Vision-RAG(related)로, "내 Vision-RAG가 도움이 되는가"를 측정한다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# plug-in 카탈로그 + 팩토리는 중앙 registries에서 가져온다.
from vision_rag_cxr.registries import (  # noqa: E402
    CATALOG_NOTES as NOTES,
    DATASET_CATALOG as DATASETS,
    ENCODER_CATALOG as ENCODERS,
    GENERATOR_CATALOG as GENERATORS,
    LABELER_CATALOG as LABELERS,
    PROMPT_CATALOG as PROMPTS,
)


# ---- ① list -----------------------------------------------------------------
def cmd_list(args):
    print("=== DATASETS (plug-in) ===")
    for k, v in DATASETS.items():
        note = f"   [주의: {NOTES[k]}]" if k in NOTES else ""
        print(f"  {k:14s} {v}{note}")
    print("\n=== GENERATORS (plug-in) ===")
    for k, v in GENERATORS.items():
        note = f"   [주의: {NOTES[k]}]" if k in NOTES else ""
        print(f"  {k:14s} {v['model_name_or_path']}{note}")
    print("\n=== ENCODERS (plug-in) ===")
    for k, v in ENCODERS.items():
        print(f"  {k:14s} {v.get('model_name_or_path', v['vision_encoder_name'])}")
    print("\n=== LABELERS (plug-in, label_space별 채점기) ===")
    for k, v in LABELERS.items():
        print(f"  {k:14s} {v}")
    print("\n=== PROMPTS (plug-in, base STYLE_PROFILE) ===")
    for k, v in PROMPTS.items():
        print(f"  {k:18s} {v}")


# ---- ② build ----------------------------------------------------------------
def cmd_build(args):
    _hf_scripts = {"indiana_hf": ("15_build_real_iu_from_hf.py", "real_iu_paired.csv")}
    if args.dataset in _hf_scripts:
        script, csv_name = _hf_scripts[args.dataset]
        cmd = [sys.executable, "-u", str(REPO_ROOT / "scripts" / script), "--out_dir", args.out, "--limit", str(args.limit)]
        print(f"build({args.dataset}):", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"canonical CSV: {Path(args.out)/csv_name}")
        return
    # 로컬/파일 기반 dataset은 dataset_registry로 dispatch
    from vision_rag_cxr.datasets.registry import preprocess_dataset
    cfg = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    cfg["dataset_type"] = args.dataset
    if args.out:
        cfg["output_dir"] = args.out
    preprocess_dataset(cfg)


# ---- ③ infer ----------------------------------------------------------------
def cmd_infer(args):
    mode_map = {"no_rag": "image_only", "image_only": "image_only", "related": "related",
                "related_oracle": "related_oracle", "unrelated": "unrelated"}
    modes = [mode_map[m.strip()] for m in args.modes.split(",")]
    gens = [g.strip() for g in args.generators.split(",")]
    for g in gens:
        if g not in GENERATORS:
            raise SystemExit(f"unknown generator: {g}. 가능: {list(GENERATORS)}")
    if args.encoder not in ENCODERS:
        raise SystemExit(f"unknown encoder: {args.encoder}. 가능: {list(ENCODERS)}")

    cfg = {
        "out_dir": args.out,
        "data_csv": args.dataset_csv,
        "label_space": args.label_space,
        "support_ratio": 0.7,
        "seeds": [args.seed],
        "top_k": args.top_k,
        "max_inference_samples": (None if args.max_samples in (0, None) else args.max_samples),
        "encoders": [ENCODERS[args.encoder]],
        "generators": [GENERATORS[g] for g in gens],
        "rag_modes": modes,
        "tasks": ["impression"],   # Vision-RAG 효과는 impression CheXbert F1로 측정
    }
    # TextGrad로 최적화된 프롬프트를 No-RAG/RAG 생성에 주입 (textgrad-first 순서)
    if getattr(args, "style_profile", None):
        cfg["optimized_style_profile_path"] = args.style_profile
    # base 프롬프트 plug-in: 카탈로그 이름 | 파일 경로 | 리터럴 텍스트 (optimized가 있으면 그쪽 우선)
    if getattr(args, "prompt_profile", None):
        cfg["style_profile"] = args.prompt_profile
    # 디바이스 주입 (auto=cuda→mps→cpu). 맥(MPS)/CPU에서도 같은 명령으로 동작.
    for gc in cfg["generators"]:
        gc["device"] = args.device
    for ec in cfg["encoders"]:
        ec["device"] = args.device
    cfg_path = Path(args.out); cfg_path.mkdir(parents=True, exist_ok=True)
    cfg_file = cfg_path / "infer_config.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    print(f"infer config: {cfg_file}")

    gpus = [g.strip() for g in str(args.gpus).split(",") if g.strip() != ""]
    if len(gpus) <= 1:
        # 단일 GPU: 직접 run_sweep
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0] if gpus else "0"
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        from vision_rag_cxr.inference.sweep import run_sweep
        cfg["_shard_id"] = 0; cfg["_num_shards"] = 1; cfg["_build_only"] = False
        # encoder config 파일 경로 주입 (retriever가 읽음)
        ecfg_dir = cfg_path / "encoder_configs"; ecfg_dir.mkdir(exist_ok=True)
        for enc in cfg["encoders"]:
            p = ecfg_dir / f"{enc.get('benchmark_tag', enc['vision_encoder_name'])}.yaml"
            p.write_text(yaml.safe_dump(enc), encoding="utf-8"); enc["_config_path"] = str(p)
        run_sweep(cfg)
    else:
        # 멀티-GPU: prebuild 후 shard 병렬 (scripts/16 직접 호출)
        print(f"multi-GPU shard across {gpus}")
        subprocess.run([sys.executable, "-u", str(REPO_ROOT/"scripts"/"16_run_experiment_sweep.py"),
                        "--config", str(cfg_file), "--build_only"],
                       env={**_env(gpus[0])}, check=False)
        procs = []
        for i, gpu in enumerate(gpus):
            log = cfg_path / f"shard_{i}.log"
            f = open(log, "w")
            procs.append(subprocess.Popen([sys.executable, "-u", str(REPO_ROOT/"scripts"/"16_run_experiment_sweep.py"),
                         "--config", str(cfg_file), "--shard_id", str(i), "--num_shards", str(len(gpus))],
                        env=_env(gpu), stdout=f, stderr=subprocess.STDOUT))
            print(f"  shard {i} -> GPU {gpu} (log {log})")
        for p in procs:
            p.wait()
    print(f"DONE. results: {args.out}/results*.csv")


def _env(gpu):
    import os
    e = dict(os.environ); e["CUDA_VISIBLE_DEVICES"] = str(gpu); e["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return e


def main():
    ap = argparse.ArgumentParser(prog="vrag", description="Vision-RAG-CXR 3-option CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="plug-in dataset/generator/encoder 목록").set_defaults(func=cmd_list)

    b = sub.add_parser("build", help="① 데이터 구축 (canonical CSV)")
    b.add_argument("--dataset", required=True, choices=list(DATASETS))
    b.add_argument("--config", default=None, help="dataset config yaml (local/padchest_gr)")
    b.add_argument("--out", default="outputs/data_build")
    b.add_argument("--limit", type=int, default=0, help="indiana_hf 샘플 수(0=전체)")
    b.set_defaults(func=cmd_build)

    f = sub.add_parser("infer", help="② inference (No-RAG vs Vision-RAG)")
    f.add_argument("--dataset-csv", required=True, help="canonical CSV (build 산출물)")
    f.add_argument("--generators", required=True, help="쉼표구분 (예: qwen2.5-vl,medgemma)")
    f.add_argument("--encoder", default="biomedclip", help=f"{list(ENCODERS)}")
    f.add_argument("--modes", default="no_rag,related", help="no_rag,related,unrelated 중 쉼표구분")
    f.add_argument("--label-space", default="chexbert_14")
    f.add_argument("--out", default="outputs/infer_run")
    f.add_argument("--max-samples", type=int, default=0, help="0=전체 inference")
    f.add_argument("--top-k", type=int, default=5)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--gpus", default="0", help="쉼표구분 CUDA GPU (예: 0,2,3). 맥/CPU면 단일값(예: 0) 또는 무시")
    f.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                   help="auto=cuda→mps(Apple)→cpu 자동. 맥북은 auto 또는 mps")
    f.add_argument("--style-profile", default=None,
                   help="TextGrad로 최적화된 STYLE_PROFILE txt 경로 (textgrad-first 순서). 지정 시 No-RAG/RAG가 이 프롬프트로 생성")
    f.add_argument("--prompt-profile", default=None,
                   help=f"base 프롬프트 plug-in: 카탈로그 이름{list(PROMPTS)} | 파일 경로 | 리터럴 텍스트")
    f.set_defaults(func=cmd_infer)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
