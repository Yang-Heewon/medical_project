"""Pydantic output schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ImpressionOutput(BaseModel):
    impression: str
    mentioned_findings: list[str] = Field(default_factory=list)
    uncertainty_phrases: list[str] = Field(default_factory=list)
    no_finding_claim: bool = False


class LesionOutput(BaseModel):
    label: str
    anatomy: str
    bbox: list[float]
    confidence: float
    evidence: str = ""


class LocalizationOutput(BaseModel):
    lesions: list[LesionOutput] = Field(default_factory=list)
    global_impression_optional: str = ""
