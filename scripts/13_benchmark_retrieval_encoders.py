#!/usr/bin/env python
"""데이터셋별 retrieval encoder 비교/자동선택 entrypoint."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from vision_rag_cxr.evaluation.encoder_benchmark import benchmark_encoders
from vision_rag_cxr.utils.io import load_yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--config", required=True, help="encoder 후보 목록 yaml")
    ap.add_argument("--out_dir", default="outputs/encoder_benchmark")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    benchmark_encoders(
        split_csv=args.split_csv,
        encoder_configs=cfg["encoders"],
        top_k=int(cfg.get("top_k", 5)),
        out_dir=args.out_dir,
        label_space=cfg.get("label_space", "chexbert_14"),
        max_inference=cfg.get("max_inference", 100),
    )

if __name__ == "__main__":
    main()
