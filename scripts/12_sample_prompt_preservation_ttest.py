#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prompt lesion-preservation sampling test.

이 스크립트는 전체 inference set을 다 돌리기 전에 작은 paired sample에서
baseline localization prompt와 candidate/preservation prompt가 병변 탐지 성능을
유의하게 바꾸는지 확인한다.

출력:
- sample_predictions.csv: uid별 baseline/candidate raw localization output
- sample_lesion_scores.csv: uid별 14-label/proxy lesion score
- sample_metric_summary.json: paired t-test, non-inferiority gate, aggregate metrics
- sample_audit.html: frontal/lateral image + bbox overlay + GT/pred label을 보는 human-loop HTML
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vision_rag_cxr.datasets.labeler_chexbert import CHEXBERT_LABELS
from vision_rag_cxr.models.generators.medgemma import MedGemmaGenerator
from vision_rag_cxr.prompting.parser import parse_json_output
from vision_rag_cxr.prompting.textgrad_optimizer import evaluate_lesion_preservation_ttest
from vision_rag_cxr.utils.io import ensure_dir, load_yaml

ABNORMAL_LABELS = [label for label in CHEXBERT_LABELS if label != "No Finding"]

LABEL_SYNONYMS = {
    "Enlarged Cardiomediastinum": ["enlarged cardiomediastinum", "wide mediastinum", "mediastinal widening"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement", "heart size enlarged"],
    "Lung Lesion": ["lung lesion", "nodule", "mass", "pulmonary nodule", "pulmonary mass"],
    "Lung Opacity": ["lung opacity", "opacity", "opacification", "infiltrate", "airspace opacity"],
    "Edema": ["edema", "pulmonary edema", "vascular congestion"],
    "Consolidation": ["consolidation", "consolidative"],
    "Pneumonia": ["pneumonia", "infectious infiltrate"],
    "Atelectasis": ["atelectasis", "atelectatic", "volume loss"],
    "Pneumothorax": ["pneumothorax", "ptx"],
    "Pleural Effusion": ["pleural effusion", "effusion", "costophrenic blunting"],
    "Pleural Other": ["pleural thickening", "pleural scarring", "pleural plaque"],
    "Fracture": ["fracture", "rib fracture", "clavicle fracture"],
    "Support Devices": ["support device", "tube", "catheter", "line", "pacemaker", "sternotomy", "port"],
}

ANATOMY_TERMS = [
    "left", "right", "bilateral", "upper", "mid", "middle", "lower", "apex", "apical", "base", "basal",
    "costophrenic", "pleural", "hilar", "hilum", "mediastinum", "cardiac", "heart", "retrocardiac",
    "lung", "lobe", "rib", "clavicle", "diaphragm", "subdiaphragmatic",
]

BASE_STYLE = """You are a careful chest X-ray assistant.
Identify visible abnormalities using only the provided frontal and lateral images.
Do not invent unsupported findings.
"""

PRESERVATION_STYLE = """You are a careful chest X-ray assistant.
Identify visible abnormalities using only the provided frontal and lateral images.
Do not invent unsupported findings.
Do not improve report wording by suppressing visible lesions.
Preserve 14-label abnormal findings and lesion/anatomy consistency.
If an abnormality is visible, keep it in the lesion list even when the final impression may be concise.
"""


def _json_loads(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    return json.loads(value)


def _clean_text(*values: Any) -> str:
    parts = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        parts.append(str(value))
    return " ".join(parts).lower()


def _gt_vector(row: pd.Series, labels: list[str]) -> np.ndarray:
    d = _json_loads(row["chexbert_labels_binary"])
    return np.asarray([int(d.get(label, 0)) for label in labels], dtype=int)


def _labels_from_prediction(pred: Any) -> set[str]:
    """VLM localization JSON을 CheXbert 14-label set으로 정규화한다."""
    if isinstance(pred, str):
        parsed, _ = parse_json_output(pred)
        pred = parsed or {"raw_text": pred}
    if not isinstance(pred, dict):
        pred = {}

    text_chunks = []
    for key in ["image_level_labels", "global_impression_optional", "raw_text"]:
        value = pred.get(key)
        if isinstance(value, list):
            text_chunks.extend(str(x) for x in value)
        elif value:
            text_chunks.append(str(value))

    lesions = pred.get("lesions", []) or []
    if isinstance(lesions, list):
        for lesion in lesions:
            if isinstance(lesion, dict):
                text_chunks.extend(str(lesion.get(k, "")) for k in ["label", "anatomy", "evidence"])
            else:
                text_chunks.append(str(lesion))

    text = " ".join(text_chunks).lower()
    labels = set()
    for label, synonyms in LABEL_SYNONYMS.items():
        if label.lower() in text or any(term in text for term in synonyms):
            labels.add(label)

    no_finding_claim = bool(pred.get("no_finding_claim", False))
    if not labels and (no_finding_claim or not lesions):
        labels.add("No Finding")
    return labels


def _pred_vector(pred: Any, labels: list[str]) -> np.ndarray:
    pred_labels = _labels_from_prediction(pred)
    abnormal = pred_labels.intersection(ABNORMAL_LABELS)
    if not abnormal:
        pred_labels.add("No Finding")
    else:
        pred_labels.discard("No Finding")
    return np.asarray([int(label in pred_labels) for label in labels], dtype=int)


def _sample_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    denom = 2 * tp + fp + fn
    return float((2 * tp) / denom) if denom else 0.0


def _sample_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return float(tp / (tp + fp)) if tp + fp else (1.0 if y_true.sum() == 0 else 0.0)


def _sample_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return float(tp / (tp + fn)) if tp + fn else 1.0


def _aggregate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = ((y_true == 1) & (y_pred == 1)).sum(axis=0).astype(float)
    fp = ((y_true == 0) & (y_pred == 1)).sum(axis=0).astype(float)
    fn = ((y_true == 1) & (y_pred == 0)).sum(axis=0).astype(float)
    tn = ((y_true == 0) & (y_pred == 0)).sum(axis=0).astype(float)

    precision_per = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall_per = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1_per = np.divide(2 * precision_per * recall_per, precision_per + recall_per, out=np.zeros_like(tp), where=(precision_per + recall_per) > 0)

    tp_micro = float(tp.sum())
    fp_micro = float(fp.sum())
    fn_micro = float(fn.sum())
    p_micro = tp_micro / (tp_micro + fp_micro) if tp_micro + fp_micro else 0.0
    r_micro = tp_micro / (tp_micro + fn_micro) if tp_micro + fn_micro else 0.0
    f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro) if p_micro + r_micro else 0.0

    return {
        "micro_precision": float(p_micro),
        "micro_recall": float(r_micro),
        "micro_f1": float(f1_micro),
        "macro_precision": float(np.mean(precision_per)),
        "macro_recall": float(np.mean(recall_per)),
        "macro_f1": float(np.mean(f1_per)),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "labelwise_accuracy": float(np.mean(y_true == y_pred)),
        "hamming_loss": float(np.mean(y_true != y_pred)),
    }


def _bbox_validity(pred: Any) -> float:
    if not isinstance(pred, dict):
        return 0.0
    lesions = pred.get("lesions", []) or []
    if not lesions:
        return 1.0
    valid = 0
    total = 0
    for lesion in lesions:
        if not isinstance(lesion, dict):
            continue
        bbox = lesion.get("bbox_norm", lesion.get("bbox"))
        if not isinstance(bbox, list) or len(bbox) != 4:
            total += 1
            continue
        try:
            vals = [float(x) for x in bbox]
        except Exception:
            total += 1
            continue
        if max(vals) > 1.5:
            # pixel bbox도 허용하되 양수/순서만 확인한다.
            ok = vals[2] > vals[0] and vals[3] > vals[1]
        else:
            ok = 0.0 <= vals[0] < vals[2] <= 1.0 and 0.0 <= vals[1] < vals[3] <= 1.0
        total += 1
        valid += int(ok)
    return float(valid / total) if total else 1.0


def _anatomy_score(row: pd.Series, pred: Any) -> float:
    expected_text = _clean_text(row.get("mesh"), row.get("problems"), row.get("impression"), row.get("findings"))
    expected = {term for term in ANATOMY_TERMS if term in expected_text}
    if not expected:
        return 1.0
    pred_text = _clean_text(json.dumps(pred, ensure_ascii=False))
    observed = {term for term in ANATOMY_TERMS if term in pred_text}
    return float(len(expected & observed) / len(expected)) if expected else 1.0


def _make_prompt(style: str) -> str:
    labels = ", ".join(CHEXBERT_LABELS)
    return f"""{style}

Task:
Given the frontal and lateral chest X-ray images, detect visible lesions and return only valid JSON.
Use this exact 14-label taxonomy when assigning labels: {labels}.

Output JSON schema:
{{
  "image_level_labels": ["<one or more labels from the 14-label taxonomy>"],
  "lesions": [
    {{
      "label": "<one label from the 14-label taxonomy>",
      "anatomy": "<short anatomical location>",
      "view": "frontal|lateral|both|unclear",
      "bbox_norm": [x1, y1, x2, y2],
      "confidence": 0.0,
      "evidence": "<short visual evidence>"
    }}
  ],
  "no_finding_claim": true/false
}}

Rules:
- bbox_norm must use normalized coordinates between 0 and 1.
- If no visible abnormality exists, use image_level_labels=["No Finding"] and lesions=[].
- Do not use labels outside the 14-label taxonomy.
"""


def _select_sample(df: pd.DataFrame, n: int, abnormal_fraction: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    abnormal_mask = []
    for x in df["chexbert_labels_binary"]:
        d = _json_loads(x)
        abnormal_mask.append(any(int(d.get(label, 0)) == 1 for label in ABNORMAL_LABELS))
    df = df.copy()
    df["_abnormal"] = abnormal_mask
    n_abn = min(int(round(n * abnormal_fraction)), int(df["_abnormal"].sum()))
    n_norm = n - n_abn
    abn = df[df["_abnormal"]]
    norm = df[~df["_abnormal"]]
    parts = []
    if n_abn:
        parts.append(abn.sample(n=n_abn, random_state=seed))
    if n_norm and len(norm):
        parts.append(norm.sample(n=min(n_norm, len(norm)), random_state=seed + 1))
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    return out.head(n)


def _image_data_uri(path: str) -> tuple[str, int, int]:
    img = Image.open(path).convert("RGB")
    width, height = img.size
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}", width, height


def _boxes_for_html(pred: dict, width: int, height: int) -> str:
    lesions = pred.get("lesions", []) if isinstance(pred, dict) else []
    chunks = []
    for lesion in lesions or []:
        if not isinstance(lesion, dict):
            continue
        bbox = lesion.get("bbox_norm", lesion.get("bbox"))
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(x) for x in bbox]
        except Exception:
            continue
        if max([x1, y1, x2, y2]) > 1.5:
            x1, x2 = x1 / width, x2 / width
            y1, y2 = y1 / height, y2 / height
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(1, x2), min(1, y2)
        label = str(lesion.get("label", "lesion"))
        chunks.append(
            f'<div class="box" style="left:{x1*100:.2f}%;top:{y1*100:.2f}%;width:{(x2-x1)*100:.2f}%;height:{(y2-y1)*100:.2f}%">'
            f'<span>{label}</span></div>'
        )
    return "\n".join(chunks)


def _write_html(rows: list[dict], out_path: Path) -> None:
    cards = []
    for row in rows:
        frontal_uri, fw, fh = _image_data_uri(row["frontal_path"])
        lateral_uri, lw, lh = _image_data_uri(row["lateral_path"])
        base_pred = json.loads(row["baseline_output_json"])
        cand_pred = json.loads(row["candidate_output_json"])
        gt_pos = ", ".join(json.loads(row["gt_positive_labels_json"]))
        base_pos = ", ".join(json.loads(row["baseline_positive_labels_json"]))
        cand_pos = ", ".join(json.loads(row["candidate_positive_labels_json"]))
        cards.append(f"""
<section class="card">
  <h2>UID {row['uid']}</h2>
  <div class="meta"><b>GT labels:</b> {gt_pos}<br><b>Baseline:</b> {base_pos}<br><b>Candidate:</b> {cand_pos}</div>
  <div class="meta"><b>Impression:</b> {row.get('gt_impression','')}</div>
  <div class="meta"><b>MeSH / Problems:</b> {row.get('mesh','')} / {row.get('problems','')}</div>
  <div class="grid">
    <div><h3>Frontal baseline</h3><div class="imgwrap"><img src="{frontal_uri}">{_boxes_for_html(base_pred, fw, fh)}</div></div>
    <div><h3>Frontal candidate</h3><div class="imgwrap"><img src="{frontal_uri}">{_boxes_for_html(cand_pred, fw, fh)}</div></div>
    <div><h3>Lateral baseline</h3><div class="imgwrap"><img src="{lateral_uri}">{_boxes_for_html(base_pred, lw, lh)}</div></div>
    <div><h3>Lateral candidate</h3><div class="imgwrap"><img src="{lateral_uri}">{_boxes_for_html(cand_pred, lw, lh)}</div></div>
  </div>
  <div class="review">
    <label><input type="checkbox"> bbox correct</label>
    <label><input type="checkbox"> lesion label correct</label>
    <label><input type="checkbox"> anatomy correct</label>
    <label><input type="checkbox"> missed lesion</label>
    <label><input type="checkbox"> hallucination</label>
    <textarea placeholder="review note"></textarea>
  </div>
</section>
""")
    html = """<!doctype html>
<html><head><meta charset="utf-8"><title>Prompt Preservation Audit</title>
<style>
body{font-family:Arial,sans-serif;margin:24px;background:#f6f7f9;color:#20242a}.card{background:white;border:1px solid #d9dee7;border-radius:8px;padding:16px;margin:0 0 20px}.grid{display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));gap:16px}.imgwrap{position:relative;border:1px solid #cbd2dd;background:#111}.imgwrap img{display:block;width:100%;height:auto}.box{position:absolute;border:3px solid #ff3b30;box-sizing:border-box}.box span{background:#ff3b30;color:white;font-size:12px;padding:2px 4px}.meta{font-size:14px;margin:8px 0}.review{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:12px}.review textarea{width:100%;height:56px}@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style></head><body>
<h1>Prompt Lesion-Preservation Sampling Audit</h1>
<p>자동 metric은 CheXbert 14-label/proxy score입니다. bbox 정답성은 사람이 overlay를 보고 확인해야 합니다.</p>
""" + "\n".join(cards) + "\n</body></html>"
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", default="outputs/production/preprocessed/indiana_paired_samples_chexbert.csv")
    parser.add_argument("--split_csv", default="outputs/production/splits/split_seed_0.csv")
    parser.add_argument("--generator_config", default="configs/models/medgemma_real.yaml")
    parser.add_argument("--out_dir", default="outputs/prompt_preservation_sample")
    parser.add_argument("--sample_size", type=int, default=8)
    parser.add_argument("--abnormal_fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    if args.split_csv and Path(args.split_csv).exists():
        df = pd.read_csv(args.split_csv)
        df = df[df["split"] == "inference"].reset_index(drop=True)
    else:
        df = pd.read_csv(args.data_csv)

    sample = _select_sample(df, args.sample_size, args.abnormal_fraction, args.seed)
    sample.to_csv(Path(out_dir) / "sampled_uids.csv", index=False)

    cfg = load_yaml(args.generator_config)
    cfg["include_context_images"] = False
    cfg["max_new_tokens"] = args.max_new_tokens
    cfg["temperature"] = 0.0
    generator = MedGemmaGenerator(cfg)

    base_prompt = _make_prompt(BASE_STYLE)
    candidate_prompt = _make_prompt(PRESERVATION_STYLE)

    pred_rows = []
    score_rows = []
    y_true_all = []
    y_base_all = []
    y_cand_all = []
    base_scores = []
    cand_scores = []

    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        print(f"[{i}/{len(sample)}] uid={row['uid']} baseline", flush=True)
        sample_dict = row.to_dict()
        base_pred = generator.generate_localization(sample_dict, base_prompt, context_examples=[])
        print(f"[{i}/{len(sample)}] uid={row['uid']} candidate", flush=True)
        cand_pred = generator.generate_localization(sample_dict, candidate_prompt, context_examples=[])

        y_true_14 = _gt_vector(row, CHEXBERT_LABELS)
        y_base_14 = _pred_vector(base_pred, CHEXBERT_LABELS)
        y_cand_14 = _pred_vector(cand_pred, CHEXBERT_LABELS)
        y_true_abn = _gt_vector(row, ABNORMAL_LABELS)
        y_base_abn = _pred_vector(base_pred, ABNORMAL_LABELS)
        y_cand_abn = _pred_vector(cand_pred, ABNORMAL_LABELS)

        base_f1 = _sample_f1(y_true_abn, y_base_abn)
        cand_f1 = _sample_f1(y_true_abn, y_cand_abn)
        base_scores.append(base_f1)
        cand_scores.append(cand_f1)
        y_true_all.append(y_true_14)
        y_base_all.append(y_base_14)
        y_cand_all.append(y_cand_14)

        gt_labels = [label for label, flag in zip(CHEXBERT_LABELS, y_true_14) if flag]
        base_labels = [label for label, flag in zip(CHEXBERT_LABELS, y_base_14) if flag]
        cand_labels = [label for label, flag in zip(CHEXBERT_LABELS, y_cand_14) if flag]

        pred_row = {
            "uid": row["uid"],
            "frontal_path": row.get("frontal_path", ""),
            "lateral_path": row.get("lateral_path", ""),
            "gt_impression": row.get("impression", ""),
            "mesh": row.get("mesh", row.get("MeSH", "")),
            "problems": row.get("problems", row.get("Problems", "")),
            "gt_positive_labels_json": json.dumps(gt_labels, ensure_ascii=False),
            "baseline_positive_labels_json": json.dumps(base_labels, ensure_ascii=False),
            "candidate_positive_labels_json": json.dumps(cand_labels, ensure_ascii=False),
            "baseline_output_json": json.dumps(base_pred, ensure_ascii=False),
            "candidate_output_json": json.dumps(cand_pred, ensure_ascii=False),
        }
        pred_rows.append(pred_row)
        score_rows.append({
            "uid": row["uid"],
            "baseline_abnormal_f1": base_f1,
            "candidate_abnormal_f1": cand_f1,
            "delta_candidate_minus_baseline": cand_f1 - base_f1,
            "baseline_abnormal_precision": _sample_precision(y_true_abn, y_base_abn),
            "candidate_abnormal_precision": _sample_precision(y_true_abn, y_cand_abn),
            "baseline_abnormal_recall": _sample_recall(y_true_abn, y_base_abn),
            "candidate_abnormal_recall": _sample_recall(y_true_abn, y_cand_abn),
            "baseline_bbox_validity": _bbox_validity(base_pred),
            "candidate_bbox_validity": _bbox_validity(cand_pred),
            "baseline_anatomy_proxy": _anatomy_score(row, base_pred),
            "candidate_anatomy_proxy": _anatomy_score(row, cand_pred),
        })

    pred_df = pd.DataFrame(pred_rows)
    score_df = pd.DataFrame(score_rows)
    pred_df.to_csv(Path(out_dir) / "sample_predictions.csv", index=False)
    score_df.to_csv(Path(out_dir) / "sample_lesion_scores.csv", index=False)

    y_true = np.vstack(y_true_all)
    y_base = np.vstack(y_base_all)
    y_cand = np.vstack(y_cand_all)
    min_samples = args.min_samples if args.min_samples is not None else len(base_scores)
    gate = evaluate_lesion_preservation_ttest(
        base_scores,
        cand_scores,
        margin=args.margin,
        alpha=args.alpha,
        mode="noninferiority",
        min_samples=min_samples,
    )
    summary = {
        "n": len(base_scores),
        "sample_size_requested": args.sample_size,
        "seed": args.seed,
        "lesion_score_for_ttest": "per-sample abnormal-label F1 excluding No Finding",
        "paired_ttest_interpretation": "p>alpha means no statistically significant paired difference was detected in this pilot; it is not proof of equivalence.",
        "baseline_14label_metrics": _aggregate_metrics(y_true, y_base),
        "candidate_14label_metrics": _aggregate_metrics(y_true, y_cand),
        "lesion_preservation_gate": gate,
    }
    (Path(out_dir) / "sample_metric_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_html(pred_rows, Path(out_dir) / "sample_audit.html")

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"saved: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
