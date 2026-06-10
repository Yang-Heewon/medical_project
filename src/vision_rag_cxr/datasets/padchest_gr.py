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


def _build_from_grounded_json(config: dict) -> pd.DataFrame:
    """실제 PadChest-GR 배포본(grounded_reports_*.json + master_table.csv)을 canonical로 변환.

    실제 스키마:
    - grounded_reports JSON = study list. 각 record: {StudyID, ImageID, findings:[...]}
        finding: {sentence_en, sentence_es, abnormal(bool), boxes:[[x1,y1,x2,y2] 정규화 0-1],
                  extra_boxes, labels:[fine label...], locations:[...], progression}
    - master_table.csv = (StudyID, ImageID, label) 행마다 label_group(coarse 26종) + split 제공.
      fine label -> label_group 매핑과 ImageID -> split을 여기서 얻는다.
      label_group 26종 = 24 finding + 'Normal'(->No Finding) + 'Other Entities'(->Other).
    """
    out_dir = ensure_dir(config.get("output_dir", "outputs/preprocessed_padchest_gr"))
    image_root = Path(config.get("image_root", "/root/research/heewon/data/padchest_gr/images"))
    verify_exists = bool(config.get("verify_image_exists", False))
    lang = str(config.get("report_lang", "en")).lower()
    include_negatives = bool(config.get("include_negatives_in_report", True))
    sent_key = "sentence_en" if lang == "en" else "sentence_es"

    grounded = Path(config["grounded_json"])
    master_table = Path(config.get("master_table", grounded.parent / "master_table.csv"))
    records = json.loads(grounded.read_text(encoding="utf-8"))

    # master_table: fine label -> label_group, ImageID -> split
    fine2group: dict[str, str] = {}
    img2split: dict[str, str] = {}
    if master_table.exists():
        mt = pd.read_csv(master_table)
        for _, mr in mt[["label", "label_group"]].dropna().iterrows():
            fine2group[str(mr["label"]).strip().lower()] = str(mr["label_group"]).strip()
        if "split" in mt.columns:
            for _, mr in mt[["ImageID", "split"]].dropna().drop_duplicates("ImageID").iterrows():
                img2split[str(mr["ImageID"])] = str(mr["split"])

    def _category(fine_label: str) -> str | None:
        """fine label -> 24 category. 'Normal'이면 None(No Finding), 그 외 못찾으면 'Other'."""
        group = fine2group.get(str(fine_label).strip().lower(), str(fine_label).strip())
        g = group.lower()
        if g == "normal":
            return None
        if g in _FINDING_LOOKUP:
            return _FINDING_LOOKUP[g]
        return "Other"

    rows = []
    n_no_image = 0
    label_counter = Counter()
    for rec in records:
        uid = str(rec.get("StudyID") or rec.get("ImageID"))
        image_name = str(rec.get("ImageID", "") or "")
        frontal_path = image_name if Path(image_name).is_absolute() else str(image_root / image_name)
        if verify_exists and not Path(frontal_path).exists():
            n_no_image += 1
            continue

        binary = {label: 0 for label in PADCHEST_GR_LABELS}
        all_sentences, pos_sentences, loc_phrases, localization_gt = [], [], [], []
        for f in rec.get("findings", []) or []:
            sent = str(f.get(sent_key) or f.get("sentence_en") or "").strip()
            if sent:
                all_sentences.append(sent)
            abnormal = bool(f.get("abnormal"))
            cats = []
            for fine in (f.get("labels") or []):
                cat = _category(fine)
                if cat is not None:
                    binary[cat] = 1
                    cats.append(cat)
            cat0 = cats[0] if cats else "Other"
            if abnormal and sent:
                pos_sentences.append(sent)
            locs = f.get("locations") or []
            if locs and cats:
                loc_phrases.append(f"{cat0}: {', '.join(str(x) for x in locs)}")
            # boxes/extra_boxes: 각각 정규화 xyxy [x1,y1,x2,y2] (reader 위치=인덱스)
            for reader_idx, box in enumerate((f.get("boxes") or []) + (f.get("extra_boxes") or [])):
                if box and len(box) >= 4:
                    localization_gt.append(
                        {"label": cat0, "bbox_xyxy": [float(v) for v in box[:4]],
                         "normalized": True, "reader": reader_idx}
                    )

        if not any(binary[l] for l in PADCHEST_GR_FINDINGS + ["Other"]):
            binary["No Finding"] = 1
        for label in PADCHEST_GR_LABELS:
            label_counter[label] += binary[label]

        report_text = " ".join(all_sentences) if include_negatives else " ".join(pos_sentences)
        rows.append({
            "uid": uid,
            "frontal_path": frontal_path,
            "lateral_path": "",
            "impression": report_text,
            "findings": " ".join(pos_sentences),
            "MeSH": "",
            "Problems": "",
            "chexbert_labels_binary": json.dumps(binary, ensure_ascii=False),
            "chexbert_labels_raw": json.dumps(binary, ensure_ascii=False),
            "anatomy_pathology_phrase": " | ".join(loc_phrases),
            "localization_gt": json.dumps(localization_gt, ensure_ascii=False),
            "split": img2split.get(image_name, ""),
            "projection_source": "padchest_gr",
        })

    paired = pd.DataFrame(rows)
    paired.to_csv(Path(out_dir) / "indiana_paired_samples.csv", index=False)  # canonical 파일명(파이프라인 호환)
    paired.to_csv(Path(out_dir) / "padchest_gr_paired.csv", index=False)
    save_jsonl((r for r in rows), Path(out_dir) / "indiana_paired_samples.jsonl")
    write_label_space_sidecar(out_dir, "padchest_gr_24")

    dist = pd.DataFrame([{"label": l, "positive_count": label_counter[l]} for l in PADCHEST_GR_LABELS])
    dist.to_csv(Path(out_dir) / "label_distribution.csv", index=False)
    with_bbox = sum(1 for r in rows if json.loads(r["localization_gt"]))
    split_counts = paired["split"].value_counts().to_dict() if len(paired) else {}
    report_md = [
        "# PadChest-GR preprocess report (real grounded_reports)",
        "",
        f"- grounded_json: {grounded}",
        f"- master_table: {master_table}",
        f"- image_root: {image_root} (verify_image_exists={verify_exists})",
        f"- report_lang: {lang}",
        f"- studies: {len(records)} / paired_samples: {len(paired)} / dropped_no_image: {n_no_image}",
        f"- with_bbox_gt: {with_bbox}",
        f"- official splits: {split_counts}",
        f"- label_space: padchest_gr_24 (No Finding + 24 findings + Other)",
        "",
        "## Label distribution (positive count)",
    ]
    report_md += [f"- {l}: {label_counter[l]}" for l in PADCHEST_GR_LABELS]
    (Path(out_dir) / "preprocess_report.md").write_text("\n".join(report_md), encoding="utf-8")
    print(f"PadChest-GR(real) paired: {len(paired)} | with_bbox={with_bbox} | splits={split_counts}", flush=True)
    return paired


def preprocess_padchest_gr(config: dict) -> pd.DataFrame:
    """PadChest-GR -> canonical paired sample table."""
    # 실제 배포본(grounded_reports JSON) 경로가 있으면 전용 빌더 사용.
    if config.get("grounded_json"):
        return _build_from_grounded_json(config)

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
