"""Experiment base class."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vision_rag_cxr.utils.io import ensure_dir


class ExperimentBase:
    """모든 실험이 공유하는 기본 기능."""

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = ensure_dir(config.get("output_dir", "outputs/experiments/default"))

    def load_inference_set(self) -> pd.DataFrame:
        df = pd.read_csv(self.config["split_csv"])
        inference = df[df["split"] == "inference"].reset_index(drop=True)
        max_samples = self.config.get("max_inference_samples")
        if max_samples is not None:
            inference = inference.head(int(max_samples)).reset_index(drop=True)
        return inference

    def save_predictions(self, rows: list[dict], filename: str = "predictions.csv") -> None:
        pd.DataFrame(rows).to_csv(Path(self.output_dir) / filename, index=False)
