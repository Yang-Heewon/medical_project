"""디바이스 자동 감지 유틸 (CUDA / Apple MPS / CPU).

CUDA가 없으면 Apple Silicon(MPS)→CPU로 자동 폴백한다. 맥북 등 비-CUDA 환경에서도
같은 코드로 돌리기 위함. 비-CUDA에서는 bitsandbytes(4bit)·device_map=auto를 쓰지 않는다.
"""

from __future__ import annotations


def resolve_device(requested: str | None = "auto") -> str:
    """'auto'면 cuda→mps→cpu 순으로 사용 가능한 디바이스를 고른다."""
    import torch

    r = str(requested or "auto").lower()
    if r == "cuda" and torch.cuda.is_available():
        return "cuda"
    if r == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if r == "cpu":
        return "cpu"
    # auto (또는 요청한 디바이스가 불가할 때)
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str, requested: str | None = None):
    """디바이스에 맞는 torch dtype. cpu=float32, mps=float16, cuda=요청(bfloat16 기본)."""
    import torch

    if device == "cpu":
        return torch.float32
    if device == "mps":
        # MPS는 bfloat16 지원이 제한적 -> float16로 통일
        return torch.float16
    name = str(requested or "bfloat16")
    return getattr(torch, name, torch.bfloat16)
