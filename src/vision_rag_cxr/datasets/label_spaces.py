"""Dataset별 label space registry.

프레임워크는 원래 CheXpert/CheXbert 14-label에 고정돼 있었지만, 데이터셋마다
선행연구/데이터셋 논문이 정의한 label scheme이 다르다. 이 모듈은 그 label space를
이름으로 등록/조회하고, 어떤 label space로 만들어진 산출물인지 sidecar로 기록한다.

등록된 space:
- ``chexbert_14``: CheXpert/CheXbert 14 labels. Indiana IU 기본.
- ``padchest_gr_24``: PadChest-GR 논문(arXiv:2411.05085)의 24개 stratification finding
  category + No Finding + Other. PadChest 174-finding ontology를 super-category로 통합한 것.

새 데이터셋을 붙일 때는 그 데이터셋 논문이 쓴 label scheme을 여기 등록하고, adapter가
그 space의 binary dict를 canonical CSV의 label 컬럼에 넣으면 된다.
"""

from __future__ import annotations

import json
from pathlib import Path

# chexbert_14는 labeler_chexbert가 정의한다(keyword labeler와 함께). 여기서 재export한다.
from vision_rag_cxr.datasets.labeler_chexbert import CHEXBERT_LABELS

# PadChest-GR (arXiv:2411.05085) Table: 24 primary finding categories used for stratification.
# 원본 PadChest 174 findings를 parent super-category로 통합, <1% prevalence는 Other로 묶음.
PADCHEST_GR_FINDINGS: list[str] = [
    "Aortic elongation",
    "Cardiomegaly",
    "Nodule",
    "Pleural effusion",
    "Scoliosis",
    "Vertebral degenerative changes",
    "Hyperinflated lung",
    "Vascular hilar enlargement",
    "Atelectasis",
    "Aortic atheromatosis",
    "Pleural thickening",
    "Interstitial pattern",
    "Alveolar pattern",
    "Electrical device",
    "Hemidiaphragm elevation",
    "Fracture",
    "Hypoexpansion",
    "Central venous catheter",
    "Hiatal hernia",
    "Endotracheal tube",
    "NSG tube",
    "Bronchiectasis",
    "Goiter",
    "Osteopenia",
]
# 정상/기타를 포함한 vector space. No Finding은 abnormal이 하나도 없을 때 1.
PADCHEST_GR_LABELS: list[str] = ["No Finding"] + PADCHEST_GR_FINDINGS + ["Other"]

LABEL_SPACES: dict[str, list[str]] = {
    "chexbert_14": list(CHEXBERT_LABELS),
    "padchest_gr_24": list(PADCHEST_GR_LABELS),
}

DEFAULT_LABEL_SPACE = "chexbert_14"
_SIDECAR_NAME = "label_space.json"


def get_label_space(name: str) -> list[str]:
    """label space 이름 -> 고정 순서 label list."""
    key = str(name or DEFAULT_LABEL_SPACE).lower()
    if key not in LABEL_SPACES:
        raise ValueError(f"등록되지 않은 label_space입니다: {name}. 가능: {list(LABEL_SPACES)}")
    return list(LABEL_SPACES[key])


def resolve_labels(config: dict | None, default: str = DEFAULT_LABEL_SPACE) -> list[str]:
    """config의 label_space 이름으로 label list를 만든다. 없으면 default."""
    name = (config or {}).get("label_space", default) if isinstance(config, dict) else default
    return get_label_space(name)


def write_label_space_sidecar(out_dir: str | Path, name: str) -> Path:
    """산출물 폴더에 어떤 label space로 만들었는지 sidecar를 남긴다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / _SIDECAR_NAME
    path.write_text(
        json.dumps({"name": name, "labels": get_label_space(name)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def read_label_space_sidecar(path_or_dir: str | Path, default: str = DEFAULT_LABEL_SPACE) -> list[str]:
    """sidecar(label_space.json)가 있으면 그 label list를, 없으면 default를 반환한다.

    파일 경로(metadata parquet 등)를 주면 같은 폴더에서 sidecar를 찾는다.
    """
    p = Path(path_or_dir)
    candidate = p / _SIDECAR_NAME if p.is_dir() else p.parent / _SIDECAR_NAME
    if candidate.exists():
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            labels = data.get("labels")
            if labels:
                return list(labels)
            if data.get("name"):
                return get_label_space(data["name"])
        except Exception:
            pass
    return get_label_space(default)
