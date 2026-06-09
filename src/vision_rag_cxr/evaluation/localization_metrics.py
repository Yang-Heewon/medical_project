"""Localization metrics.

세 가지 평가 경로를 지원한다.

1. 기하학적 IoU/mAP (``iou_localization_metrics``):
   ``localization_gt``(bbox GT가 있는 데이터셋, 예: PadChest-GR)와 model bbox를 label+IoU로 매칭.
2. Claude/human grounding audit (``build_grounding_audit_sheet`` + ``summarize_audit``):
   bbox를 이미지에 오버레이해 저장하고, Claude Code(또는 사람)가 각 박스의 label 매핑이
   맞는지 판정해 CSV(`claude_correct`/`human_correct`)를 채우면 그것을 요약한다.
3. (legacy) ``summarize_human_audit``.

IU-only처럼 bbox GT가 없는 데이터셋은 1번을 쓰지 말고 2번(audit)으로 보고한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------- #
# bbox geometry
# --------------------------------------------------------------------------- #
def iou(box_a, box_b) -> float:
    """두 bbox(xyxy)의 IoU."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return float(inter / max(area_a + area_b - inter, 1e-8))


def _parse_gt(value) -> list[dict]:
    """localization_gt 컬럼 -> [{label, bbox_xyxy}]."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except Exception:
            return []
    return []


def _parse_pred_lesions(value) -> list[dict]:
    """model localization 출력 -> [{label, bbox}]. lesions 리스트나 JSON 문자열 모두 허용."""
    if isinstance(value, str) and value.strip():
        try:
            value = json.loads(value)
        except Exception:
            return []
    if isinstance(value, dict):
        value = value.get("lesions", [])
    out = []
    for les in value or []:
        box = les.get("bbox") or les.get("bbox_xyxy")
        if box and len(box) >= 4:
            out.append({"label": les.get("label", ""), "bbox_xyxy": [float(v) for v in box[:4]]})
    return out


def iou_localization_metrics(
    pred_by_uid: dict[str, list[dict]],
    gt_by_uid: dict[str, list[dict]],
    iou_thr: float = 0.5,
    label_aware: bool = True,
) -> dict:
    """예측 bbox vs GT bbox를 label+IoU로 greedy 매칭해 detection metric을 계산한다.

    반환: precision, recall, f1, mean_iou(matched), per_label recall, n_pred, n_gt, n_match.
    """
    n_pred = n_gt = n_match = 0
    matched_ious = []
    per_label_gt: dict[str, int] = {}
    per_label_hit: dict[str, int] = {}

    for uid, gts in gt_by_uid.items():
        preds = list(pred_by_uid.get(uid, []))
        n_gt += len(gts)
        n_pred += len(preds)
        used = set()
        for g in gts:
            per_label_gt[g["label"]] = per_label_gt.get(g["label"], 0) + 1
            best_iou, best_j = 0.0, -1
            for j, p in enumerate(preds):
                if j in used:
                    continue
                if label_aware and str(p.get("label", "")).lower() != str(g["label"]).lower():
                    continue
                v = iou(p["bbox_xyxy"], g["bbox_xyxy"])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0 and best_iou >= iou_thr:
                used.add(best_j)
                n_match += 1
                matched_ious.append(best_iou)
                per_label_hit[g["label"]] = per_label_hit.get(g["label"], 0) + 1

    precision = n_match / n_pred if n_pred else 0.0
    recall = n_match / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "iou_threshold": iou_thr,
        "label_aware": label_aware,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "n_match": n_match,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou_matched": round(sum(matched_ious) / len(matched_ious), 4) if matched_ious else 0.0,
        "per_label_recall": {
            lbl: round(per_label_hit.get(lbl, 0) / cnt, 4) for lbl, cnt in sorted(per_label_gt.items())
        },
    }


# --------------------------------------------------------------------------- #
# Claude / human grounding audit
# --------------------------------------------------------------------------- #
def build_grounding_audit_sheet(
    paired_csv: str,
    out_dir: str,
    box_source: str = "gt",
    gt_column: str = "localization_gt",
    pred_csv: str | None = None,
    max_samples: int | None = 50,
    box_color: str = "red",
) -> str:
    """각 finding bbox를 이미지에 오버레이해 저장하고 audit 시트(CSV)를 만든다.

    box_source:
    - ``gt``  : paired_csv의 localization_gt(GT bbox)를 검수
    - ``pred``: pred_csv의 model lesions(예측 bbox)를 검수

    생성된 ``grounding_audit.csv``의 ``claude_correct``/``claude_notes`` 칸을
    Claude Code(또는 사람)가 overlay 이미지를 보고 채운 뒤 ``summarize_audit``로 요약한다.
    """
    from PIL import Image, ImageDraw

    out = Path(out_dir)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(paired_csv)
    if max_samples is not None:
        df = df.head(int(max_samples))

    pred_lookup: dict[str, list[dict]] = {}
    if box_source == "pred":
        if not pred_csv:
            raise ValueError("box_source='pred'면 pred_csv가 필요합니다.")
        pdf = pd.read_csv(pred_csv)
        pred_col = "parsed_output" if "parsed_output" in pdf.columns else "raw_output"
        for _, pr in pdf.iterrows():
            pred_lookup[str(pr["uid"])] = _parse_pred_lesions(pr.get(pred_col))

    rows = []
    for _, r in df.iterrows():
        uid = str(r["uid"])
        frontal = r.get("frontal_path")
        if not frontal or not Path(str(frontal)).exists():
            continue
        findings = (
            _parse_gt(r.get(gt_column)) if box_source == "gt" else pred_lookup.get(uid, [])
        )
        for i, f in enumerate(findings):
            box = f.get("bbox_xyxy")
            if not box:
                continue
            img = Image.open(str(frontal)).convert("RGB")
            draw = ImageDraw.Draw(img)
            draw.rectangle([float(box[0]), float(box[1]), float(box[2]), float(box[3])], outline=box_color, width=3)
            draw.text((float(box[0]) + 2, max(0, float(box[1]) - 10)), str(f.get("label", "")), fill=box_color)
            overlay_path = out / "overlays" / f"{uid}_{i}_{box_source}.png"
            img.save(overlay_path)
            rows.append(
                {
                    "uid": uid,
                    "finding_id": i,
                    "label": f.get("label", ""),
                    "bbox_xyxy": json.dumps([float(v) for v in box[:4]]),
                    "box_source": box_source,
                    "overlay_path": str(overlay_path),
                    "claude_correct": "",   # Claude Code/사람이 채움: 1(맞음)/0(틀림)
                    "claude_notes": "",
                }
            )

    audit_csv = out / "grounding_audit.csv"
    pd.DataFrame(rows).to_csv(audit_csv, index=False)
    print(f"grounding audit sheet: {audit_csv} ({len(rows)} boxes, source={box_source})", flush=True)
    return str(audit_csv)


def summarize_audit(audit_csv: str) -> dict:
    """audit CSV(claude_correct 또는 human_correct)에서 correctness를 요약한다."""
    df = pd.read_csv(audit_csv)
    col = "claude_correct" if "claude_correct" in df.columns else "human_correct"
    scored = df[pd.to_numeric(df[col], errors="coerce").notna()].copy()
    scored[col] = pd.to_numeric(scored[col])
    per_label = (
        scored.groupby("label")[col].mean().round(4).to_dict() if "label" in scored.columns else {}
    )
    return {
        "audit_count": int(len(df)),
        "scored_count": int(len(scored)),
        "box_correctness_rate": round(float(scored[col].mean()), 4) if len(scored) else 0.0,
        "per_label_correctness": per_label,
        "verifier_column": col,
    }


def summarize_human_audit(audit_csv: str) -> dict:
    """(legacy) human_correct 기반 요약. 신규 코드는 summarize_audit 사용."""
    df = pd.read_csv(audit_csv)
    col = "human_correct" if "human_correct" in df.columns else "claude_correct"
    return {
        "audit_count": int(len(df)),
        "box_correctness_rate": float(pd.to_numeric(df[col], errors="coerce").mean()) if len(df) else 0.0,
    }
