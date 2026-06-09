"""Dataset 전처리 dispatcher.

새 데이터셋을 붙이려면 canonical schema를 뱉는 preprocess 함수를 작성하고 여기 등록하면 된다.
canonical schema (모든 adapter가 지켜야 하는 컬럼):
    uid, frontal_path, lateral_path, impression, findings,
    chexbert_labels_binary(JSON: label->0/1, 해당 데이터셋 label space 기준),
    chexbert_labels_raw, anatomy_pathology_phrase
선택 컬럼: localization_gt(JSON, bbox GT가 있는 데이터셋), projection_source

config의 ``dataset_type`` (또는 ``dataset``)으로 어떤 adapter를 쓸지 고른다. 기본은 indiana.
"""

from __future__ import annotations

from vision_rag_cxr.datasets.indiana import preprocess_indiana
from vision_rag_cxr.datasets.padchest_gr import preprocess_padchest_gr

PREPROCESSORS = {
    "indiana": preprocess_indiana,
    "indiana_iu_cxr": preprocess_indiana,
    "iu": preprocess_indiana,
    "padchest_gr": preprocess_padchest_gr,
    "padchest-gr": preprocess_padchest_gr,
}


def preprocess_dataset(config: dict):
    """config['dataset_type']에 맞는 adapter로 canonical table을 만든다."""
    name = str(config.get("dataset_type") or config.get("dataset") or "indiana").lower()
    if name not in PREPROCESSORS:
        raise ValueError(f"등록되지 않은 dataset_type입니다: {name}. 가능: {sorted(set(PREPROCESSORS))}")
    return PREPROCESSORS[name](config)
