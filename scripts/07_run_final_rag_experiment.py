from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 로컬에서 python scripts/*.py를 실행해도 src 패키지를 찾을 수 있게 한다.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vision_rag_cxr.inference.experiments.final_rag_experiment import FinalRAGExperiment
from vision_rag_cxr.utils.io import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    gen_cfg = load_yaml(cfg["generator_config"])
    FinalRAGExperiment(cfg).run(gen_cfg)


if __name__ == "__main__":
    main()
