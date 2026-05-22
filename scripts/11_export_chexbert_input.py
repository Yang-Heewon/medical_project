#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd


def safe(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired_csv", default="outputs/preprocessed/indiana_paired_samples.csv")
    parser.add_argument("--out_csv", default="outputs/preprocessed/chexbert_input_reports.csv")
    parser.add_argument("--source", default="impression", choices=["impression", "findings_impression"])
    args = parser.parse_args()

    df = pd.read_csv(args.paired_csv)

    rows = []
    for _, r in df.iterrows():
        impression = safe(r.get("impression", ""))
        findings = safe(r.get("findings", ""))

        if args.source == "impression":
            report = impression if impression else (findings + "\n" + impression).strip()
        else:
            report = (findings + "\n" + impression).strip()

        # /root/CheXbert/run_chexbert.py는 id/report column을 기대한다.
        # uid도 같이 보존해두면 다른 CheXbert wrapper와의 호환성이 좋아진다.
        rows.append({
            "id": r["uid"],
            "uid": r["uid"],
            "report": report,
        })

    out = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("rows:", len(out))


if __name__ == "__main__":
    main()
