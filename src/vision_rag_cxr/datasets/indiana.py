"""Indiana University Chest X-ray canonical 전처리.

정책 (docs/PROJECT_FRAMEWORK_PROMPT_KO.md 2절):
- split unit은 uid.
- frontal + lateral image가 모두 있는 uid만 사용한다.
- ``indiana_projections.csv`` 같은 projection metadata가 있으면 우선 사용한다.
- projection metadata가 없으면 uid prefix image list를 정렬해 fallback pair를 만든다.
- raw data는 절대 덮어쓰지 않고 outputs/preprocessed 아래에 canonical table을 저장한다.

입력 (raddar Kaggle 스키마):
- report_csv 컬럼: uid, MeSH, Problems, image, indication, comparison, findings, impression
- projection_csv 컬럼: uid, filename, projection (projection ∈ {Frontal, Lateral})

출력 (outputs/preprocessed):
- indiana_paired_samples.csv / .jsonl
- label_distribution.csv
- preprocess_report.md

생성 컬럼:
- uid, frontal_path, lateral_path, impression, findings, MeSH, Problems,
  chexbert_labels_binary(JSON), chexbert_labels_raw(JSON), anatomy_pathology_phrase,
  projection_source
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from vision_rag_cxr.datasets.labeler_chexbert import CHEXBERT_LABELS, CheXbertLikeLabeler
from vision_rag_cxr.utils.io import ensure_dir, save_jsonl

_FRONTAL_TOKENS = ("frontal", "pa", "ap", "front")
_LATERAL_TOKENS = ("lateral", "lat", "decub")


def _classify_projection(value: str) -> str | None:
    """projection 문자열을 frontal/lateral로 정규화한다."""
    v = str(value or "").strip().lower()
    if not v:
        return None
    if any(tok in v for tok in _LATERAL_TOKENS):
        return "lateral"
    if any(tok in v for tok in _FRONTAL_TOKENS):
        return "frontal"
    return None


def _resolve_projection_csv(cfg: dict, report_csv: Path) -> Path | None:
    explicit = cfg.get("projection_csv")
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    sibling = report_csv.parent / "indiana_projections.csv"
    if sibling.exists():
        return sibling
    return None


def _pairs_from_projections(
    projection_csv: Path, image_root: Path, verify_exists: bool
) -> dict[str, dict[str, str]]:
    """projection metadata로 uid -> {frontal_path, lateral_path}를 만든다."""
    proj = pd.read_csv(projection_csv)
    cols = {c.lower(): c for c in proj.columns}
    uid_col = cols.get("uid", "uid")
    file_col = cols.get("filename", cols.get("image", "filename"))
    proj_col = cols.get("projection", "projection")

    pairs: dict[str, dict[str, str]] = {}
    for _, r in proj.iterrows():
        uid = str(r[uid_col])
        view = _classify_projection(r[proj_col])
        if view is None:
            continue
        path = image_root / str(r[file_col])
        if verify_exists and not path.exists():
            continue
        slot = "frontal_path" if view == "frontal" else "lateral_path"
        pairs.setdefault(uid, {})
        # 같은 view가 여러 장이면 첫 번째만 쓴다.
        pairs[uid].setdefault(slot, str(path))
    return pairs


def _pairs_from_filename_fallback(
    image_root: Path, uids: list[str]
) -> dict[str, dict[str, str]]:
    """projection metadata가 없을 때 uid prefix image list를 정렬해 pair를 만든다."""
    all_images = sorted(p for p in image_root.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    by_uid: dict[str, list[Path]] = {}
    for p in all_images:
        # raddar filename 예: "1_IM-0001-3001.dcm.png" -> uid prefix "1"
        prefix = p.name.split("_", 1)[0]
        by_uid.setdefault(prefix, []).append(p)

    pairs: dict[str, dict[str, str]] = {}
    for uid in uids:
        imgs = sorted(by_uid.get(str(uid), []))
        if len(imgs) >= 2:
            pairs[str(uid)] = {"frontal_path": str(imgs[0]), "lateral_path": str(imgs[1])}
        elif len(imgs) == 1:
            pairs[str(uid)] = {"frontal_path": str(imgs[0])}
    return pairs


def preprocess_indiana(config: dict) -> pd.DataFrame:
    """canonical paired sample table을 생성/저장하고 DataFrame을 반환한다."""
    image_root = Path(config.get("image_root") or config.get("image_dir") or "/root/data/images/images_normalized")
    report_csv = Path(config.get("report_csv") or "/root/data/label/indiana_reports.csv")
    out_dir = ensure_dir(config.get("output_dir", "outputs/preprocessed"))
    require_both = bool(config.get("require_frontal_lateral", True))
    verify_exists = bool(config.get("verify_image_exists", True))
    min_impression_chars = int(config.get("min_impression_chars", 0))

    if not report_csv.exists():
        raise FileNotFoundError(f"report_csv not found: {report_csv}")

    reports = pd.read_csv(report_csv)
    # 컬럼 이름을 대소문자/별칭에 관대하게 매핑한다.
    cols = {c.lower(): c for c in reports.columns}

    def col(name: str, *alts: str) -> str | None:
        for cand in (name, *alts):
            if cand.lower() in cols:
                return cols[cand.lower()]
        return None

    uid_col = col("uid")
    if uid_col is None:
        raise ValueError(f"report_csv에 uid 컬럼이 없습니다. columns={list(reports.columns)}")
    impression_col = col("impression")
    findings_col = col("findings")
    mesh_col = col("MeSH", "mesh")
    problems_col = col("Problems", "problems")

    uids = [str(u) for u in reports[uid_col].tolist()]

    # pairing
    projection_csv = _resolve_projection_csv(config, report_csv)
    if projection_csv is not None:
        pairs = _pairs_from_projections(projection_csv, image_root, verify_exists)
        projection_source = f"projection_csv:{projection_csv.name}"
    else:
        pairs = _pairs_from_filename_fallback(image_root, uids)
        projection_source = "filename_fallback"

    labeler = CheXbertLikeLabeler(config.get("labeling", {}))

    rows = []
    n_no_pair = 0
    for _, r in reports.iterrows():
        uid = str(r[uid_col])
        pair = pairs.get(uid, {})
        frontal = pair.get("frontal_path")
        lateral = pair.get("lateral_path")
        if frontal is None or (require_both and lateral is None):
            n_no_pair += 1
            continue

        impression = "" if impression_col is None else str(r.get(impression_col, "") or "")
        findings = "" if findings_col is None else str(r.get(findings_col, "") or "")
        if len(impression.strip()) < min_impression_chars and len((findings + impression).strip()) < min_impression_chars:
            continue

        mesh = "" if mesh_col is None else str(r.get(mesh_col, "") or "")
        problems = "" if problems_col is None else str(r.get(problems_col, "") or "")

        binary, raw = labeler.label_report(impression, findings)
        phrase = " | ".join(x for x in [mesh, problems, impression[:300]] if x.strip())

        rows.append(
            {
                "uid": uid,
                "frontal_path": frontal,
                "lateral_path": lateral or "",
                "impression": impression,
                "findings": findings,
                "MeSH": mesh,
                "Problems": problems,
                "chexbert_labels_binary": json.dumps(binary, ensure_ascii=False),
                "chexbert_labels_raw": json.dumps(raw, ensure_ascii=False),
                "anatomy_pathology_phrase": phrase,
                "projection_source": projection_source,
            }
        )

    paired = pd.DataFrame(rows)
    paired_csv = Path(out_dir) / "indiana_paired_samples.csv"
    paired.to_csv(paired_csv, index=False)
    save_jsonl((r for r in rows), Path(out_dir) / "indiana_paired_samples.jsonl")

    # label distribution
    label_counter = Counter()
    for r in rows:
        binary = json.loads(r["chexbert_labels_binary"])
        for label in CHEXBERT_LABELS:
            label_counter[label] += int(binary.get(label, 0))
    dist = pd.DataFrame(
        [{"label": label, "positive_count": label_counter[label]} for label in CHEXBERT_LABELS]
    )
    dist.to_csv(Path(out_dir) / "label_distribution.csv", index=False)

    report_md = [
        "# Indiana preprocess report",
        "",
        f"- report_csv: {report_csv}",
        f"- image_root: {image_root}",
        f"- projection_source: {projection_source}",
        f"- total_reports: {len(reports)}",
        f"- paired_samples: {len(paired)}",
        f"- dropped_no_pair: {n_no_pair}",
        f"- require_frontal_lateral: {require_both}",
        "",
        "## Label distribution (positive count)",
    ]
    report_md += [f"- {label}: {label_counter[label]}" for label in CHEXBERT_LABELS]
    if len(paired) == 0:
        report_md += [
            "",
            "## WARNING",
            "paired_samples=0. image_root/projection_csv 경로와 파일 존재 여부를 확인하세요.",
        ]
    (Path(out_dir) / "preprocess_report.md").write_text("\n".join(report_md), encoding="utf-8")

    print(f"paired samples: {len(paired)} (dropped_no_pair={n_no_pair}, source={projection_source})", flush=True)
    return paired
