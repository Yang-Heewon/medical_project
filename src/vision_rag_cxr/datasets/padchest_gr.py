"""PadChest-GR (arXiv:2411.05085) canonical 전처리 adapter.

PadChest-GR은 grounded radiology report 데이터셋이다. study마다:
- positive finding 문장(영/스) + 각 문장의 24-category finding label + location + bbox(최대 2 reader)
- negative finding 문장(병변 부재 진술)

이 adapter는 PadChest-GR record를 Indiana와 동일한 canonical schema로 변환한다.
label space는 ``padchest_gr_24`` (PadChest-GR 논문 stratification scheme).
추가로 bbox GT를 ``localization_gt`` 컬럼(JSON)에 담아 실제 IoU localization 평가를 가능하게 한다.

배포 파일의 실제 컬럼명은 gated(BIMCV) 배포본에서 확정되므로, config의 column override로
매핑한다. 기본값은 논문에서 추정한 논리 스키마다.

기대 입력(설정 가능):
- master_csv 또는 master_jsonl: study 단위 record
  - id_column            : study/image uid
  - image_column         : image 파일명(또는 절대경로)
  - findings_column      : positive finding record JSON list
      각 record: {label, sentence_en, sentence_es, locations, boxes}
      boxes: [[x, y, w, h or x2, y2], ...]
  - negatives_column     : negative finding 문장 JSON list (optional)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from vision_rag_cxr.datasets.label_spaces import (
    PADCHEST_GR_FINDINGS,
    PADCHEST_GR_LABELS,
    write_label_space_sidecar,
)
from vision_rag_cxr.utils.io import ensure_dir, save_jsonl

# 24 finding category의 소문자 정규화 lookup.
_FINDING_LOOKUP = {f.lower(): f for f in PADCHEST_GR_FINDINGS}


def _normalize_finding(raw_label: str, extra_map: dict[str, str]) -> str:
    """raw finding label을 24 category 중 하나로 정규화한다. 못 찾으면 'Other'."""
    if raw_label is None:
        return "Other"
    key = str(raw_label).strip().lower()
    if key in _FINDING_LOOKUP:
        return _FINDING_LOOKUP[key]
    if key in extra_map:
        mapped = extra_map[key]
        return _FINDING_LOOKUP.get(str(mapped).lower(), "Other")
    return "Other"


def _to_xyxy(box, fmt: str) -> list[float]:
    """box를 [x1,y1,x2,y2]로 정규화한다. fmt: 'xywh' 또는 'xyxy'."""
    b = [float(v) for v in box[:4]]
    if fmt == "xywh":
        x, y, w, h = b
        return [x, y, x + w, y + h]
    return b


def _parse_findings(value):
    """findings 컬럼 값을 list[dict]로 파싱한다."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def preprocess_padchest_gr(config: dict) -> pd.DataFrame:
    """PadChest-GR -> canonical paired sample table."""
    image_root = Path(config.get("image_root", "/root/data/padchest_gr/images"))
    master = config.get("master_csv") or config.get("master_jsonl")
    if not master:
        raise ValueError("padchest_gr config에 master_csv 또는 master_jsonl 경로가 필요합니다.")
    master = Path(master)
    if not master.exists():
        raise FileNotFoundError(f"PadChest-GR master file not found: {master}")

    out_dir = ensure_dir(config.get("output_dir", "outputs/preprocessed_padchest_gr"))
    verify_exists = bool(config.get("verify_image_exists", True))
    lang = str(config.get("report_lang", "en")).lower()
    include_negatives = bool(config.get("include_negatives_in_report", True))
    bbox_fmt = str(config.get("bbox_format", "xywh")).lower()
    extra_map = {str(k).lower(): v for k, v in (config.get("finding_label_map", {}) or {}).items()}

    cols = config.get("columns", {})
    id_col = cols.get("id_column", "StudyID")
    image_col = cols.get("image_column", "ImageID")
    findings_col = cols.get("findings_column", "findings")
    negatives_col = cols.get("negatives_column", "negative_findings")

    if master.suffix == ".jsonl":
        df = pd.DataFrame([json.loads(l) for l in master.read_text(encoding="utf-8").splitlines() if l.strip()])
    elif master.suffix == ".parquet":
        df = pd.read_parquet(master)
    else:
        df = pd.read_csv(master)

    sent_key = "sentence_en" if lang == "en" else "sentence_es"
    rows = []
    n_no_image = 0
    label_counter = Counter()

    for _, r in df.iterrows():
        uid = str(r.get(id_col))
        image_name = str(r.get(image_col, "") or "")
        frontal_path = image_name if Path(image_name).is_absolute() else str(image_root / image_name)
        if verify_exists and not Path(frontal_path).exists():
            n_no_image += 1
            continue

        findings = _parse_findings(r.get(findings_col))
        pos_sentences = []
        loc_phrases = []
        localization_gt = []
        binary = {label: 0 for label in PADCHEST_GR_LABELS}

        for f in findings:
            cat = _normalize_finding(f.get("label"), extra_map)
            binary[cat] = 1
            sent = str(f.get(sent_key) or f.get("sentence_en") or "").strip()
            if sent:
                pos_sentences.append(sent)
            locs = f.get("locations") or []
            if locs:
                loc_phrases.append(f"{cat}: {', '.join(str(x) for x in locs)}")
            for reader_idx, box_set in enumerate(f.get("boxes") or []):
                # box_set은 단일 box [x,y,w,h] 또는 box list일 수 있다.
                box_list = box_set if (box_set and isinstance(box_set[0], (list, tuple))) else [box_set]
                for box in box_list:
                    if box:
                        localization_gt.append(
                            {"label": cat, "bbox_xyxy": _to_xyxy(box, bbox_fmt), "reader": reader_idx}
                        )

        # No Finding: abnormal finding이 하나도 없으면 1.
        if not any(binary[l] for l in PADCHEST_GR_FINDINGS + ["Other"]):
            binary["No Finding"] = 1

        neg_sentences = []
        if include_negatives:
            negs = r.get(negatives_col)
            if isinstance(negs, str) and negs.strip():
                try:
                    negs = json.loads(negs)
                except Exception:
                    negs = [negs]
            if isinstance(negs, list):
                neg_sentences = [str(x) for x in negs if str(x).strip()]

        findings_text = " ".join(pos_sentences)
        report_text = " ".join(pos_sentences + (neg_sentences if include_negatives else []))

        for label in PADCHEST_GR_LABELS:
            label_counter[label] += binary[label]

        rows.append(
            {
                "uid": uid,
                "frontal_path": frontal_path,
                "lateral_path": "",
                "impression": report_text,
                "findings": findings_text,
                "MeSH": "",
                "Problems": "",
                "chexbert_labels_binary": json.dumps(binary, ensure_ascii=False),
                "chexbert_labels_raw": json.dumps(binary, ensure_ascii=False),
                "anatomy_pathology_phrase": " | ".join(loc_phrases),
                "localization_gt": json.dumps(localization_gt, ensure_ascii=False),
                "projection_source": "padchest_gr",
            }
        )

    paired = pd.DataFrame(rows)
    paired.to_csv(Path(out_dir) / "indiana_paired_samples.csv", index=False)  # canonical 파일명 유지(파이프라인 호환)
    save_jsonl((r for r in rows), Path(out_dir) / "indiana_paired_samples.jsonl")
    write_label_space_sidecar(out_dir, "padchest_gr_24")

    dist = pd.DataFrame(
        [{"label": label, "positive_count": label_counter[label]} for label in PADCHEST_GR_LABELS]
    )
    dist.to_csv(Path(out_dir) / "label_distribution.csv", index=False)

    report_md = [
        "# PadChest-GR preprocess report",
        "",
        f"- master: {master}",
        f"- image_root: {image_root}",
        f"- report_lang: {lang}",
        f"- total_records: {len(df)}",
        f"- paired_samples: {len(paired)}",
        f"- dropped_no_image: {n_no_image}",
        f"- label_space: padchest_gr_24",
        f"- with_bbox_gt: {sum(1 for r in rows if json.loads(r['localization_gt']))}",
        "",
        "## Label distribution (positive count)",
    ]
    report_md += [f"- {label}: {label_counter[label]}" for label in PADCHEST_GR_LABELS]
    (Path(out_dir) / "preprocess_report.md").write_text("\n".join(report_md), encoding="utf-8")

    print(f"PadChest-GR paired samples: {len(paired)} (dropped_no_image={n_no_image}, label_space=padchest_gr_24)", flush=True)
    return paired
