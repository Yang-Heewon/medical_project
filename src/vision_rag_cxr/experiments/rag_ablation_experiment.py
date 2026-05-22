"""RAG ablation experiment skeleton.

1/3/5 localization, 2/4/6 impression 비교를 위한 공통 class 위치.
현재 template에서는 retriever 연결 위치를 명확히 잡아두고,
실제 관련/무관 retrieval은 support_metadata + FAISS search 로 확장한다.
"""

from __future__ import annotations

from vision_rag_cxr.experiments.impression_experiment import ImpressionExperiment
from vision_rag_cxr.experiments.localization_experiment import LocalizationExperiment


class RAGAblationExperiment:
    def __init__(self, config: dict):
        self.config = config

    def run(self, generator_config: dict):
        if self.config["task_type"] == "impression":
            return ImpressionExperiment(self.config).run(generator_config)
        if self.config["task_type"] == "localization":
            return LocalizationExperiment(self.config).run(generator_config)
        raise ValueError(f"Unknown task_type: {self.config['task_type']}")
