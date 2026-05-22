"""간단한 registry 패턴.

새 모델/metric/retriever를 추가할 때 if-else를 늘리지 않기 위해 사용한다.
"""

from __future__ import annotations


class Registry:
    """문자열 이름으로 class/function을 찾아 쓰기 위한 registry."""

    def __init__(self, name: str):
        self.name = name
        self._items = {}

    def register(self, key: str):
        def deco(obj):
            self._items[key] = obj
            return obj
        return deco

    def get(self, key: str):
        if key not in self._items:
            raise KeyError(f"{self.name} registry에 '{key}'가 없습니다. 사용 가능: {list(self._items)}")
        return self._items[key]

    def keys(self):
        return list(self._items.keys())


MODEL_REGISTRY = Registry("model")
ENCODER_REGISTRY = Registry("encoder")
RETRIEVER_REGISTRY = Registry("retriever")
METRIC_REGISTRY = Registry("metric")
EXPERIMENT_REGISTRY = Registry("experiment")
