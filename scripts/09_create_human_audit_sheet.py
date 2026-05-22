#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFile, ImageFont


ImageFile.LOAD_TRUNCATED_IMAGES = True

CHEXPERT_LABELS = [
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

LOCALIZABLE_LABELS = {
    "Lung Lesion",
    "Lung Opacity",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
}


def safe_name(value):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9가-힣_.-]+", "_", value)
    return value.strip("_") or "unknown"


def load_json(value, default):
    try:
        return json.loads(value)
    except Exception:
        return default


def gt_positive_labels(gt_json):
    labels = load_json(gt_json, {})
    out = []
    for label, value in labels.items():
        try:
            if int(value) == 1:
                out.append(label)
        except Exception:
            continue
    return out


def gt_has_any_abnormal(gt_json):
    return any(label != "No Finding" for label in gt_positive_labels(gt_json))


def gt_has_localizable_abnormal(gt_json):
    return any(label in LOCALIZABLE_LABELS for label in gt_positive_labels(gt_json))


def parsed_lesions(parsed_json):
    obj = load_json(parsed_json, {})
    lesions = obj.get("lesions", []) if isinstance(obj, dict) else []
    return [x for x in lesions if isinstance(x, dict)]


def pred_has_bbox(parsed_json):
    return len(parsed_lesions(parsed_json)) > 0


def pred_chexpert_labels(parsed_json):
    labels = set()
    for lesion in parsed_lesions(parsed_json):
        label = str(lesion.get("chexpert_label", "")).strip()
        if label in CHEXPERT_LABELS and label != "No Finding":
            labels.add(label)
    return labels


def binary_metrics(gt_values, pred_values):
    tp = fp = fn = tn = 0
    for gt, pred in zip(gt_values, pred_values):
        gt = bool(gt)
        pred = bool(pred)
        if gt and pred:
            tp += 1
        elif not gt and pred:
            fp += 1
        elif gt and not pred:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "support": tp + fn,
        "pred_positive": tp + fp,
    }


def compute_weak_metrics(df):
    gt_any = [gt_has_any_abnormal(x) for x in df["gt_labels"]]
    gt_loc = [gt_has_localizable_abnormal(x) for x in df["gt_labels"]]
    pred_bbox = [pred_has_bbox(x) for x in df["parsed_json"]]

    case_rows = []
    for name, gt_values in [
        ("any_abnormal_gt_vs_any_bbox", gt_any),
        ("localizable_gt_vs_any_bbox", gt_loc),
    ]:
        row = {"metric": name, "num_cases": len(df)}
        row.update(binary_metrics(gt_values, pred_bbox))
        case_rows.append(row)

    per_label_rows = []
    pred_label_sets = [pred_chexpert_labels(x) for x in df["parsed_json"]]
    gt_label_sets = [set(gt_positive_labels(x)) for x in df["gt_labels"]]
    for label in CHEXPERT_LABELS:
        if label == "No Finding":
            continue
        gt_values = [label in labels for labels in gt_label_sets]
        pred_values = [label in labels for labels in pred_label_sets]
        row = {"label": label}
        row.update(binary_metrics(gt_values, pred_values))
        per_label_rows.append(row)

    summary = {
        "num_cases": int(len(df)),
        "runtime_errors": int((df["error"].astype(str) != "").sum()),
        "parse_errors": int((df["parse_error"].astype(str) != "").sum()),
        "cases_with_bbox": int(sum(pred_bbox)),
        "cases_without_bbox": int(len(df) - sum(pred_bbox)),
        "total_bboxes": int(sum(len(parsed_lesions(x)) for x in df["parsed_json"])),
        "weak_case_metrics": case_rows,
    }

    label_counter = Counter()
    for parsed_json in df["parsed_json"]:
        for lesion in parsed_lesions(parsed_json):
            label = lesion.get("label") or lesion.get("chexpert_label") or "abnormality"
            label_counter[str(label)] += 1
    summary["top_predicted_free_text_labels"] = dict(label_counter.most_common(30))

    return summary, pd.DataFrame(case_rows), pd.DataFrame(per_label_rows)


def sample_cases(df, n, seed):
    if n < 0 or n >= len(df):
        return df.copy().reset_index(drop=True)

    work = df.copy()
    work["_gt_localizable"] = work["gt_labels"].apply(gt_has_localizable_abnormal)
    work["_pred_bbox"] = work["parsed_json"].apply(pred_has_bbox)
    work["_group"] = work.apply(
        lambda r: f"gt{int(r['_gt_localizable'])}_pred{int(r['_pred_bbox'])}", axis=1
    )

    groups = ["gt1_pred1", "gt0_pred1", "gt1_pred0", "gt0_pred0"]
    base = max(n // len(groups), 1)
    sampled = []
    used = set()

    for group in groups:
        part = work[work["_group"] == group]
        if part.empty:
            continue
        take = min(base, len(part), n - len(sampled))
        if take <= 0:
            break
        chosen = part.sample(n=take, random_state=seed + len(sampled))
        sampled.append(chosen)
        used.update(chosen.index.tolist())

    out = pd.concat(sampled, ignore_index=False) if sampled else work.iloc[[]]
    if len(out) < n:
        rest = work.drop(index=list(used), errors="ignore")
        take = min(n - len(out), len(rest))
        if take > 0:
            out = pd.concat([out, rest.sample(n=take, random_state=seed + 99)], ignore_index=False)

    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def wrap_text(text, width=95):
    words = str(text).replace("\n", " ").split()
    lines = []
    cur = ""
    for word in words:
        if len(cur) + len(word) + 1 <= width:
            cur = (cur + " " + word).strip()
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_overlay(row, out_path, max_width=900):
    img = Image.open(row["frontal_path"]).convert("RGB")
    w, h = img.size
    scale = max_width / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)))

    draw = ImageDraw.Draw(img)
    iw, ih = img.size
    font = ImageFont.load_default()
    colors = ["red", "yellow", "cyan", "lime", "magenta", "orange"]

    lesions = parsed_lesions(row.get("parsed_json", "{}"))
    for idx, lesion in enumerate(lesions):
        bbox = lesion.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        x1, x2 = int(float(x1) * iw), int(float(x2) * iw)
        y1, y2 = int(float(y1) * ih), int(float(y2) * ih)
        color = colors[idx % len(colors)]
        for t in range(4):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)
        label = lesion.get("label") or lesion.get("chexpert_label") or "abnormality"
        draw.text((x1 + 4, max(0, y1 - 16)), f"{idx}: {label}", fill=color, font=font)

    panel_h = 330
    canvas = Image.new("RGB", (iw, ih + panel_h), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    gt_labels = ", ".join(gt_positive_labels(row.get("gt_labels", "{}"))) or "none"
    pred_labels = []
    for idx, lesion in enumerate(lesions):
        label = lesion.get("label") or lesion.get("chexpert_label") or "abnormality"
        pred_labels.append(f"{idx}: {label} bbox={lesion.get('bbox')} conf={lesion.get('confidence')}")

    lines = [
        f"UID: {row.get('uid', '')}",
        f"GT labels: {gt_labels}",
        f"Predicted boxes: {len(lesions)}",
        f"Parse error: {row.get('parse_error', '')}",
        f"Runtime error: {row.get('error', '')}",
        "Predictions:",
    ]
    lines.extend(pred_labels or ["none"])
    lines.append(f"GT impression: {row.get('gt_impression', '')}")
    lines.append(f"GT findings: {row.get('gt_findings', '')}")

    y = ih + 10
    for line in lines:
        for subline in wrap_text(line):
            draw.text((10, y), subline, fill="black", font=font)
            y += 15
            if y > ih + panel_h - 15:
                break
        if y > ih + panel_h - 15:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def audit_rows_for_case(row, rel_image_path):
    gt_labels = ";".join(gt_positive_labels(row.get("gt_labels", "{}")))
    lesions = parsed_lesions(row.get("parsed_json", "{}"))
    base = {
        "uid": row.get("uid", ""),
        "visualization_path": rel_image_path,
        "frontal_path": row.get("frontal_path", ""),
        "lateral_path": row.get("lateral_path", ""),
        "gt_positive_labels": gt_labels,
        "gt_impression": row.get("gt_impression", ""),
        "gt_findings": row.get("gt_findings", ""),
        "parse_error": row.get("parse_error", ""),
        "raw_output": row.get("raw_output", ""),
    }

    rows = []
    if not lesions:
        item = dict(base)
        item.update({
            "lesion_id": "",
            "predicted_label": "",
            "predicted_chexpert_label": "",
            "predicted_anatomy": "",
            "predicted_laterality": "",
            "predicted_bbox": "",
            "predicted_confidence": "",
            "human_bbox_correct": "",
            "human_label_correct": "",
            "human_clinically_relevant": "",
            "human_missed_visible_lesion": "",
            "human_score_0_05_1": "",
            "human_notes": "",
        })
        rows.append(item)
        return rows

    for lesion_id, lesion in enumerate(lesions):
        item = dict(base)
        item.update({
            "lesion_id": lesion_id,
            "predicted_label": lesion.get("label", ""),
            "predicted_chexpert_label": lesion.get("chexpert_label", ""),
            "predicted_anatomy": lesion.get("anatomy", ""),
            "predicted_laterality": lesion.get("laterality", ""),
            "predicted_bbox": json.dumps(lesion.get("bbox", ""), ensure_ascii=False),
            "predicted_confidence": lesion.get("confidence", ""),
            "human_bbox_correct": "",
            "human_label_correct": "",
            "human_clinically_relevant": "",
            "human_missed_visible_lesion": "",
            "human_score_0_05_1": "",
            "human_notes": "",
        })
        rows.append(item)
    return rows


def write_html(out_dir, case_records, summary):
    rows = []
    for rec in case_records:
        row = rec["row"]
        image_rel = html.escape(rec["image_rel"])
        uid = html.escape(str(row.get("uid", "")))
        gt = html.escape(", ".join(gt_positive_labels(row.get("gt_labels", "{}"))) or "none")
        lesions = parsed_lesions(row.get("parsed_json", "{}"))
        pred_lines = []
        for idx, lesion in enumerate(lesions):
            label = html.escape(str(lesion.get("label") or lesion.get("chexpert_label") or "abnormality"))
            bbox = html.escape(str(lesion.get("bbox", "")))
            pred_lines.append(f"<li>{idx}: {label} bbox={bbox}</li>")
        if not pred_lines:
            pred_lines.append("<li>no predicted bbox</li>")

        rows.append(
            f'<section class="card">'
            f'<a href="{image_rel}"><img src="{image_rel}" loading="lazy"></a>'
            f'<div class="meta"><b>UID:</b> {uid}<br><b>GT:</b> {gt}<br>'
            f'<b>Pred boxes:</b> {len(lesions)}<br>'
            f'<b>Parse error:</b> {html.escape(str(row.get("parse_error", "")))}</div>'
            f'<ul>{"".join(pred_lines)}</ul>'
            f'</section>'
        )

    case_metrics = summary.get("weak_case_metrics", [])
    metric_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(m['metric']))}</td>"
        f"<td>{m['accuracy']}</td><td>{m['precision']}</td><td>{m['recall']}</td><td>{m['f1']}</td>"
        f"<td>{m['tp']}</td><td>{m['fp']}</td><td>{m['fn']}</td><td>{m['tn']}</td>"
        "</tr>"
        for m in case_metrics
    )

    body = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MedGemma Lesion BBox Audit</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; background: #f6f7f9; color: #111; }}
    h1 {{ margin-bottom: 4px; }}
    .summary {{ background: white; border: 1px solid #ddd; padding: 12px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; }}
    th, td {{ border: 1px solid #ccc; padding: 4px 7px; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 14px; }}
    .card {{ background: white; border: 1px solid #ddd; padding: 10px; }}
    .card img {{ width: 100%; border: 1px solid #aaa; }}
    .meta, li {{ font-size: 13px; line-height: 1.35; }}
  </style>
</head>
<body>
  <h1>MedGemma Lesion BBox Audit</h1>
  <div class="summary">
    <p><b>Cases:</b> {summary['num_cases']} &nbsp; <b>Cases with bbox:</b> {summary['cases_with_bbox']} &nbsp;
    <b>Total bboxes:</b> {summary['total_bboxes']} &nbsp; <b>Parse errors:</b> {summary['parse_errors']}</p>
    <p>Automatic metrics below are weak CheXbert-label metrics, not bbox IoU. Human audit is required for true localization correctness.</p>
    <table>
      <tr><th>metric</th><th>acc</th><th>precision</th><th>recall</th><th>f1</th><th>tp</th><th>fp</th><th>fn</th><th>tn</th></tr>
      {metric_rows}
    </table>
  </div>
  <div class="grid">
    {''.join(rows)}
  </div>
</body>
</html>
"""
    html_path = out_dir / "index.html"
    html_path.write_text(body, encoding="utf-8")
    return html_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection_csv", required=True)
    parser.add_argument("--out_dir", default="outputs/human_audit/medgemma_lesion_bbox_audit")
    parser.add_argument("--n", type=int, default=120, help="Number of cases for human audit. Use -1 for all cases.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_image_width", type=int, default=900)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.detection_csv, keep_default_na=False)
    summary, case_metrics, per_label_metrics = compute_weak_metrics(df)
    sample_df = sample_cases(df, args.n, args.seed)

    audit_rows = []
    case_records = []
    for _, row in sample_df.iterrows():
        uid = safe_name(row.get("uid", "unknown"))
        out_img = image_dir / f"uid_{uid}.jpg"
        draw_overlay(row, out_img, max_width=args.max_image_width)
        rel = out_img.relative_to(out_dir).as_posix()
        case_records.append({"row": row, "image_rel": rel})
        audit_rows.extend(audit_rows_for_case(row, rel))

    audit_csv = out_dir / "audit_lesions.csv"
    pd.DataFrame(audit_rows).to_csv(audit_csv, index=False)

    case_metrics.to_csv(out_dir / "weak_case_metrics.csv", index=False)
    per_label_metrics.to_csv(out_dir / "weak_per_label_metrics.csv", index=False)
    (out_dir / "weak_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path = write_html(out_dir, case_records, summary)

    print("Saved HTML:", html_path)
    print("Saved audit CSV:", audit_csv)
    print("Saved weak metrics:", out_dir / "weak_case_metrics.csv")
    print("Saved per-label weak metrics:", out_dir / "weak_per_label_metrics.csv")
    print("audit cases:", len(sample_df))
    print("audit rows:", len(audit_rows))
    print("cases_with_bbox:", summary["cases_with_bbox"])
    print("total_bboxes:", summary["total_bboxes"])
    print("parse_errors:", summary["parse_errors"])
    print(case_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
