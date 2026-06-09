"""HF ykumards/open-i 스트리밍으로 실제 Indiana CXR canonical 데이터 구축 (무인증).

별도 Kaggle 키 없이 실제 IU(frontal+lateral 이미지 + report + MeSH/Problems)를 받는다.
canonical schema(uid, frontal_path, lateral_path, impression, findings, MeSH, Problems,
chexbert_labels_binary, chexbert_labels_raw, anatomy_pathology_phrase)로 저장한다.

주의: HF `datasets` 스트리밍은 인터프리터 종료 시 GIL 관련 crash를 낼 수 있다. CLI/스크립트에서
호출할 때는 별도 프로세스에서 돌리고 끝나면 os._exit(0)로 빠지는 것을 권장(scripts/15 참고).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from vision_rag_cxr.datasets.labeler_chexbert import CheXbertLikeLabeler
from vision_rag_cxr.utils.io import ensure_dir


def build_indiana_from_hf(out_dir: str, limit: int = 0, hf_name: str = "ykumards/open-i") -> int:
    """HF 스트리밍으로 IU를 받아 <out_dir>/real_iu_paired.csv + images/*.png 생성. 샘플 수 반환."""
    from datasets import load_dataset

    out = ensure_dir(out_dir)
    img = ensure_dir(Path(out) / "images")
    lab = CheXbertLikeLabeler()
    ds = load_dataset(hf_name, split="train", streaming=True)
    rows: list[dict] = []
    for ex in ds:
        if limit and len(rows) >= limit:
            break
        try:
            f = Image.open(io.BytesIO(ex["img_frontal"])).convert("L")
            l = Image.open(io.BytesIO(ex["img_lateral"])).convert("L")
        except Exception:
            continue
        uid = str(ex["uid"])
        fp = img / f"{uid}_F.png"
        lp = img / f"{uid}_L.png"
        f.save(fp)
        l.save(lp)
        imp = str(ex.get("impression") or "")
        fnd = str(ex.get("findings") or "")
        binary, raw = lab.label_report(imp, fnd)
        rows.append({
            "uid": uid, "frontal_path": str(fp), "lateral_path": str(lp),
            "impression": imp, "findings": fnd,
            "MeSH": str(ex.get("MeSH") or ""), "Problems": str(ex.get("Problems") or ""),
            "chexbert_labels_binary": json.dumps(binary, ensure_ascii=False),
            "chexbert_labels_raw": json.dumps(raw, ensure_ascii=False),
            "anatomy_pathology_phrase": (str(ex.get("MeSH") or "") + " | " + imp[:200]),
            "projection_source": "hf_open_i",
        })
        if len(rows) % 200 == 0:
            pd.DataFrame(rows).to_csv(Path(out) / "real_iu_paired.csv", index=False)
            print(f"... {len(rows)} samples", flush=True)

    pd.DataFrame(rows).to_csv(Path(out) / "real_iu_paired.csv", index=False)
    (Path(out) / "BUILD_DONE").write_text(f"{len(rows)}\n", encoding="utf-8")
    print(f"DONE real IU samples: {len(rows)} -> {Path(out)/'real_iu_paired.csv'}", flush=True)
    return len(rows)
