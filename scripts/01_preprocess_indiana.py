#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Indiana University Chest X-ray 전처리 entrypoint.

config-driven src pipeline을 호출한다. 이전 standalone 로직은
``vision_rag_cxr.data.indiana_preprocessor``로 모았다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vision_rag_cxr.data.indiana_preprocessor import preprocess_indiana
from vision_rag_cxr.utils.io import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data/indiana.yaml")
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--report_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config) if args.config else {}
    if args.image_root:
        cfg["image_root"] = args.image_root
    if args.report_csv:
        cfg["report_csv"] = args.report_csv
    if args.out_dir:
        cfg["output_dir"] = args.out_dir

    out = preprocess_indiana(cfg)
    print("paired samples:", len(out))
    print("output_dir:", cfg.get("output_dir", "outputs/preprocessed"))


if __name__ == "__main__":
    main()
