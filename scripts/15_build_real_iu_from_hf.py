#!/usr/bin/env python
"""thin wrapper -> vision_rag_cxr.datasets.indiana_hf.build_indiana_from_hf

HF datasets 스트리밍 종료 시 GIL crash 회피를 위해 끝나면 os._exit(0)로 빠진다.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision_rag_cxr.datasets.indiana_hf import build_indiana_from_hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/root/data/real_iu")
    ap.add_argument("--limit", type=int, default=0, help="0=전체")
    args = ap.parse_args()
    build_indiana_from_hf(args.out_dir, args.limit)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
