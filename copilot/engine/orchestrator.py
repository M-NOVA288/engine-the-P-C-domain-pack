"""Runs a domain pack as a fixed graph of steps with a human-approval gate.

    intake -> analysis -> rules (RAG) -> scoring -> decide -> memo -> [human gate]

The graph is deliberately explicit. Swapping this for LangGraph or Microsoft
Agent Framework is a drop-in change: each step is already a pure
`CaseFile -> CaseFile` function, which maps one-to-one onto graph nodes.
"""
from __future__ import annotations

import time
from pathlib import Path

from .audit import AuditLog
from .llm import MEMO_SYSTEM_PROMPT, get_llm
from .pack import DomainPack
from .retrieval import get_retriever
from .schema import CaseFile, Decision


class Orchestrator:
    def __init__(self, pack: DomainPack, audit_path: Path | str = "audit/audit.jsonl") -> None:
        self.pack = pack
        self.retriever = get_retriever(pack.guideline_path)
        self.llm = get_llm()
        self.audit = AuditLog(audit_path)

    def _timed(self, case: CaseFile, name: str, fn) -> CaseFile:
        t0 = time.perf_counter()
        case = fn(case)
        case.timings_ms[name] = round((time.perf_counter() - t0) * 1000, 2)
        self.audit.record(case.case_id, name, {
            "findings": [f.model_dump() for f in case.findings if f.step == name],
            "ms": case.timings_ms[name],
        })
        return case

    def run(self, case_id: str, documents: dict[str, str]) -> CaseFile:
        case = CaseFile(case_id=case_id, pack=self.pack.name, documents=documents)
        case = self._timed(case, "intake", self.pack.extract)
        case = self._timed(case, "analysis", self.pack.analyze)
        case = self._timed(case, "rules", lambda c: self.pack.check_rules(c, self.retriever))
        case = self._timed(case, "scoring", self.pack.score)
        case = self._timed(case, "decide", self.pack.decide)

        t0 = time.perf_counter()
        case.memo_markdown = self.llm.write(MEMO_SYSTEM_PROMPT, self.pack.draft_memo(case))
        case.timings_ms["memo"] = round((time.perf_counter() - t0) * 1000, 2)
        case.timings_ms["total"] = round(sum(case.timings_ms.values()), 2)

        self.audit.record(case_id, "recommendation", {
            "recommendation": case.recommendation, "risk_score": case.risk_score,
        })
        return case

    def approve(self, case: CaseFile, reviewer: str, decision: Decision, note: str = "") -> CaseFile:
        """Human-in-the-loop: nothing is bound or declined until a person signs off."""
        case.human_decision = decision
        case.human_reviewer = reviewer
        self.audit.record(case.case_id, "human_decision", {
            "reviewer": reviewer, "decision": decision,
            "agreed_with_copilot": decision == case.recommendation, "note": note,
        })
        return case
