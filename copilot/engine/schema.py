"""Domain-agnostic data contracts shared by every domain pack.

Every finding carries a citation so a reviewer can trace each statement in the
memo back to a source document line or a guideline section.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Decision(str, Enum):
    ACCEPT = "accept"
    REFER = "refer"      # send to a human with reasons; the safe default under uncertainty
    DECLINE = "decline"


class Citation(BaseModel):
    source: str                 # e.g. "application.txt" or "guidelines.md#3.2"
    locator: str = ""           # line number, section id, or field name
    snippet: str = ""           # short excerpt shown to the reviewer


class Finding(BaseModel):
    step: str                   # which agent produced it: intake, appetite, loss, scoring
    severity: str = "info"      # info | warn | block
    message: str
    citations: list[Citation] = Field(default_factory=list)


class ExtractedField(BaseModel):
    name: str
    value: Any = None
    confidence: float = 1.0     # 0..1; low confidence or missing value drives a referral
    citation: Citation | None = None


class CaseFile(BaseModel):
    """Everything known about one case as it moves through the pipeline."""
    case_id: str
    pack: str
    documents: dict[str, str]                       # filename -> raw text
    fields: dict[str, ExtractedField] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    risk_score: float | None = None
    risk_drivers: list[dict[str, float | str]] = Field(default_factory=list)
    recommendation: Decision | None = None
    memo_markdown: str = ""
    human_decision: Decision | None = None
    human_reviewer: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)

    def value(self, name: str, default: Any = None) -> Any:
        f = self.fields.get(name)
        return default if f is None or f.value is None else f.value
