"""A domain pack plugs industry knowledge into the shared engine.

To add a new industry (lending, prior auth, claims), implement these steps and
supply a guidelines document. The orchestrator, audit log, retrieval, eval
harness and ROI calculator are reused unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .schema import CaseFile


class DomainPack(ABC):
    name: str
    guideline_path: Path
    # Fields that must be present with confidence >= threshold or the case is referred.
    required_fields: list[str] = []
    min_confidence: float = 0.8

    @abstractmethod
    def extract(self, case: CaseFile) -> CaseFile: ...

    @abstractmethod
    def check_rules(self, case: CaseFile, retriever) -> CaseFile: ...

    @abstractmethod
    def analyze(self, case: CaseFile) -> CaseFile: ...

    @abstractmethod
    def score(self, case: CaseFile) -> CaseFile: ...

    @abstractmethod
    def decide(self, case: CaseFile) -> CaseFile: ...

    @abstractmethod
    def draft_memo(self, case: CaseFile) -> str: ...
