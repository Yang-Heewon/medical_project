"""ROCO (Radiology Objects in COntext) — 멀티-모달리티 radiology image-caption 데이터셋.

HF `eltorio/ROCOv2-radiology`(무인증) 스트리밍. chest가 아닌 모달리티(복부·뇌 등)도 포함하므로
프레임워크의 dataset-불문/비-chest robustness를 실제로 테스트한다.

ROCO는 caption만 있고 CheXpert 라벨이 없다 → impression=caption, 라벨은 비움({}).
TextGrad는 '텍스트(impression 스타일)' 목표로 동작하고, 평가도 impression 텍스트(BERTScore/ROUGE) 위주.
(병변 라벨 기반 지표/게이트는 ROCO에 미적용 — label space 무관)
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from vision_rag_cxr.utils.io import ensure_dir


def build_roco_from_hf(out_dir: str, limit: int = 0, hf_name: str = "eltorio/ROCOv2-radiology",
                       split: str = "train") -> int:
    """ROCO를 스트리밍으로 받아 canonical CSV(impression=caption, 라벨 없음) 생성. 샘플 수 반환."""
    from datasets import load_dataset

    out = ensure_dir(out_dir)
    img = ensure_dir(Path(out) / "images")
    ds = load_dataset(hf_name, split=split, streaming=True)
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        if limit and len(rows) >= limit:
            break
        im = ex.get("image")
        try:
            if isinstance(im, dict) and "bytes" in im:
                im = Image.open(io.BytesIO(im["bytes"]))
            im = im.convert("L")
        except Exception:
            continue
        uid = str(ex.get("image_id", i))
        fp = img / f"{uid}.png"
        try:
            im.save(fp)
        except Exception:
            continue
        cap = str(ex.get("caption") or "").strip()
        if not cap:
            continue
        rows.append({
            "uid": uid, "frontal_path": str(fp), "lateral_path": "",
            "impression": cap, "findings": cap, "MeSH": "", "Problems": "",
            "chexbert_labels_binary": "{}",          # ROCO: 라벨 없음 (텍스트 전용)
            "chexbert_labels_raw": "{}",
            "anatomy_pathology_phrase": cap[:200],
            "projection_source": "roco", "modality": "radiology (mixed modality)",
        })
        if len(rows) % 200 == 0:
            pd.DataFrame(rows).to_csv(Path(out) / "roco_paired.csv", index=False)
            print(f"... {len(rows)} samples", flush=True)

    pd.DataFrame(rows).to_csv(Path(out) / "roco_paired.csv", index=False)
    (Path(out) / "BUILD_DONE").write_text(f"{len(rows)}\n", encoding="utf-8")
    print(f"DONE ROCO samples: {len(rows)} -> {Path(out)/'roco_paired.csv'}", flush=True)
    return len(rows)
