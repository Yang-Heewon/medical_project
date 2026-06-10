#!/usr/bin/env python
"""thin wrapper -> vision_rag_cxr.datasets.roco.build_roco_from_hf (datasets 종료 GIL crash 회피용 os._exit)"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision_rag_cxr.datasets.roco import build_roco_from_hf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/root/research/heewon/data/roco")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hf_name", default="eltorio/ROCOv2-radiology")
    args = ap.parse_args()
    build_roco_from_hf(args.out_dir, args.limit, args.hf_name)
    sys.stdout.flush(); os._exit(0)

if __name__ == "__main__":
    main()
