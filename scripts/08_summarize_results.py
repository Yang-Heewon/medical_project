from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 로컬에서 python scripts/*.py를 실행해도 src 패키지를 찾을 수 있게 한다.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import glob
import pandas as pd
from vision_rag_cxr.utils.io import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="outputs/experiments")
    parser.add_argument("--out_dir", default="outputs/summary")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    files = glob.glob(str(Path(args.results_root) / "**" / "predictions.csv"), recursive=True)

    rows = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f
        rows.append(df)

    if rows:
        all_df = pd.concat(rows, ignore_index=True)
        all_df.to_csv(Path(out_dir) / "main_results.csv", index=False)
    else:
        pd.DataFrame().to_csv(Path(out_dir) / "main_results.csv", index=False)

    (Path(out_dir) / "final_report.md").write_text(
        "# Final report\n\nGenerated summary placeholder. Add metric computation after connecting production models.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
