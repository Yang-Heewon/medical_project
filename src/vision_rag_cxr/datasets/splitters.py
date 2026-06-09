"""Support / Inference multilabel stratified split.

정책 (docs/PROJECT_FRAMEWORK_PROMPT_KO.md 4절):
- split unit은 uid. 동일 uid가 support/inference에 동시에 들어가면 안 된다.
- support:inference = 7:3 기본.
- multilabel iterative stratification으로 label drift를 낮춘다.
- seed별 split과 split quality(label prevalence drift)를 저장한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vision_rag_cxr.datasets.labeler_chexbert import CHEXBERT_LABELS
from vision_rag_cxr.datasets.label_spaces import resolve_labels
from vision_rag_cxr.utils.io import ensure_dir


def _label_matrix(df: pd.DataFrame, labels: list[str] | None = None, column: str = "chexbert_labels_binary") -> np.ndarray:
    """label JSON 컬럼을 고정 순서 0/1 matrix로 변환한다."""
    labels = labels or CHEXBERT_LABELS
    rows = []
    for value in df[column]:
        d = json.loads(value) if isinstance(value, str) else (value or {})
        rows.append([int(d.get(label, 0)) for label in labels])
    return np.asarray(rows, dtype=int)


def multilabel_split(df: pd.DataFrame, support_ratio: float, seed: int = 0, labels: list[str] | None = None) -> pd.DataFrame:
    """df에 'split' 컬럼('support'/'inference')을 추가해 반환한다.

    iterative stratification을 우선 쓰고, 실패하면 deterministic shuffle split으로 fallback한다.
    어느 경우든 support/inference 둘 다 비어 있지 않도록 보장한다.
    """
    n = len(df)
    if n < 2:
        raise ValueError(f"split하기에 sample이 너무 적습니다: n={n}")

    test_size = max(1, min(n - 1, int(round(n * (1.0 - support_ratio)))))
    y = _label_matrix(df, labels)
    indices = np.arange(n)
    support_idx = inference_idx = None

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size / n, random_state=seed
        )
        support_idx, inference_idx = next(msss.split(indices.reshape(-1, 1), y))
    except Exception:
        # rare label/degenerate matrix 등으로 iterstrat가 실패하면 random shuffle로 fallback.
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        inference_idx = perm[:test_size]
        support_idx = perm[test_size:]

    out = df.copy().reset_index(drop=True)
    split = np.empty(n, dtype=object)
    split[support_idx] = "support"
    split[inference_idx] = "inference"
    out["split"] = split
    return out


def _split_quality(df: pd.DataFrame, labels: list[str] | None = None) -> pd.DataFrame:
    """label별 support/inference prevalence와 drift를 요약한다."""
    labels = labels or CHEXBERT_LABELS
    y = _label_matrix(df, labels)
    split = df["split"].to_numpy()
    sup_mask = split == "support"
    inf_mask = split == "inference"
    rows = []
    for j, label in enumerate(labels):
        col = y[:, j]
        sup_prev = float(col[sup_mask].mean()) if sup_mask.any() else 0.0
        inf_prev = float(col[inf_mask].mean()) if inf_mask.any() else 0.0
        rows.append(
            {
                "label": label,
                "total_positive": int(col.sum()),
                "support_positive": int(col[sup_mask].sum()),
                "inference_positive": int(col[inf_mask].sum()),
                "support_prevalence": round(sup_prev, 4),
                "inference_prevalence": round(inf_prev, 4),
                "prevalence_drift": round(abs(sup_prev - inf_prev), 4),
            }
        )
    return pd.DataFrame(rows)


def create_splits(data_csv: str, config: dict) -> dict:
    """data_csv를 읽어 seed별 split/quality/summary를 output_dir에 저장한다."""
    out_dir = ensure_dir(config.get("output_dir", "outputs/splits"))
    support_ratio = float(config.get("support_ratio", 0.7))
    labels = resolve_labels(config)
    seeds = config.get("seeds", [config.get("seed", 0)])
    if isinstance(seeds, int):
        seeds = [seeds]

    df = pd.read_csv(data_csv)
    if "chexbert_labels_binary" not in df.columns:
        raise ValueError(
            "split 입력 CSV에 'chexbert_labels_binary' 컬럼이 없습니다. "
            "먼저 preprocess_indiana로 canonical table을 생성하세요."
        )

    drift_warn = float(config.get("max_prevalence_drift_warning", 0.05))
    min_pos_warn = int(config.get("min_positive_count_warning", 5))
    summary_lines = ["# Split summary", "", f"- support_ratio: {support_ratio}", f"- seeds: {seeds}", ""]
    written = {}

    for seed in seeds:
        seed = int(seed)
        split_df = multilabel_split(df, support_ratio, seed=seed, labels=labels)
        split_path = Path(out_dir) / f"split_seed_{seed}.csv"
        split_df.to_csv(split_path, index=False)

        quality = _split_quality(split_df, labels)
        quality_path = Path(out_dir) / f"split_quality_seed_{seed}.csv"
        quality.to_csv(quality_path, index=False)
        written[seed] = str(split_path)

        n_sup = int((split_df["split"] == "support").sum())
        n_inf = int((split_df["split"] == "inference").sum())
        max_drift = float(quality["prevalence_drift"].max()) if len(quality) else 0.0
        rare = quality[quality["total_positive"] < min_pos_warn]["label"].tolist()
        summary_lines.extend(
            [
                f"## seed {seed}",
                f"- support: {n_sup}, inference: {n_inf}",
                f"- max_prevalence_drift: {round(max_drift, 4)}"
                + ("  (WARNING: exceeds threshold)" if max_drift > drift_warn else ""),
                f"- rare_labels(<{min_pos_warn} positives): {rare if rare else 'none'}",
                "",
            ]
        )

    (Path(out_dir) / "split_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"splits written: {written}", flush=True)
    return written
