#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Support / Inference split 생성 entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vision_rag_cxr.datasets.splitters import create_splits
from vision_rag_cxr.utils.io import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/split/stratified_70_30.yaml")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--support_ratio", type=float, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config) if args.config else {}
    if args.out_dir:
        cfg["output_dir"] = args.out_dir
    if args.support_ratio is not None:
        cfg["support_ratio"] = args.support_ratio
    if args.seeds:
        cfg["seeds"] = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    create_splits(args.data_csv, cfg)
    print("output_dir:", cfg.get("output_dir", "outputs/splits"))


if __name__ == "__main__":
    main()
