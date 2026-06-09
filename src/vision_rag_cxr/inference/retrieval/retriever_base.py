"""Retriever base class."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query_sample: dict, top_k: int) -> list[dict]:
        pass
