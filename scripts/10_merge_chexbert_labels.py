#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import pandas as pd


LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def apply_uncertain_policy(v, policy: str):
    """
    CheXbert raw value를 binary label로 변환한다.

    보통:
    1 = positive
    0 = negative
    -1 = uncertain

    u_one이면 uncertain도 positive로 둔다.
    """
    if pd.isna(v):
        return 0

    try:
        v = int(float(v))
    except Exception:
        text = str(v).strip().lower()
        if text in ["1", "positive", "pos", "present"]:
            return 1
        if text in ["-1", "uncertain", "u"]:
            v = -1
        else:
            return 0

    if v == 1:
        return 1

    if v == -1:
        return 1 if policy == "u_one" else 0

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired_csv", required=True)
    parser.add_argument("--chexbert_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--uncertain_policy", default="u_one")
    args = parser.parse_args()

    paired = pd.read_csv(args.paired_csv)
    labels = pd.read_csv(args.chexbert_csv)

    if "uid" not in labels.columns:
        if "id" in labels.columns:
            labels = labels.rename(columns={"id": "uid"})
        else:
            raise ValueError("chexbert_csv must have uid or id column")

    missing = [label for label in LABELS if label not in labels.columns]
    if missing:
        raise ValueError(f"Missing label columns: {missing}")

    paired["uid"] = paired["uid"].astype(str)
    labels["uid"] = labels["uid"].astype(str)

    raw_map = {}
    binary_map = {}

    for _, row in labels.iterrows():
        uid = str(row["uid"])

        raw = {}
        binary = {}

        for label in LABELS:
            raw_value = row[label]
            raw[label] = None if pd.isna(raw_value) else int(raw_value)
            binary[label] = apply_uncertain_policy(raw_value, args.uncertain_policy)

        raw_map[uid] = json.dumps(raw, ensure_ascii=False)
        binary_map[uid] = json.dumps(binary, ensure_ascii=False)

    paired["chexbert_labels_raw"] = paired["uid"].map(raw_map)
    paired["chexbert_labels_binary"] = paired["uid"].map(binary_map)

    before = len(paired)
    paired = paired.dropna(subset=["chexbert_labels_binary"]).reset_index(drop=True)
    after = len(paired)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("before:", before)
    print("after:", after)
    print("dropped_no_chexbert:", before - after)
    print("uncertain_policy:", args.uncertain_policy)


if __name__ == "__main__":
    main()
