#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Support set 기반 RAG DB 구축 entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vision_rag_cxr.inference.retrieval.build_database import build_support_database
from vision_rag_cxr.utils.io import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_csv", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/retrieval/hybrid_rag.yaml")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config) if args.config else {}
    if args.out_dir:
        cfg["output_dir"] = args.out_dir

    build_support_database(args.split_csv, cfg)
    print("output_dir:", cfg.get("output_dir", "outputs/rag"))


if __name__ == "__main__":
    main()
