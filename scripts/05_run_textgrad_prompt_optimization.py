from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 로컬에서 python scripts/*.py를 실행해도 src 패키지를 찾을 수 있게 한다.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vision_rag_cxr.experiments.prompt_optimization_experiment import run_prompt_optimization
from vision_rag_cxr.utils.io import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    run_prompt_optimization(load_yaml(args.config))


if __name__ == "__main__":
    main()
