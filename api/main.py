"""HTTP API (FastAPI). Deploy to Azure Container Apps; see infra/README.md.

    uvicorn api.main:app --reload
    curl -X POST localhost:8000/cases -H 'content-type: application/json' \
         -d '{"case_id": "DEMO-1", "documents": {"application.txt": "...", "sov.txt": "..."}}'
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from copilot.engine import CaseFile, Decision, Orchestrator
from copilot.packs import get_pack

app = FastAPI(title="Risk Decision Copilot", version="0.1.0")
_orchestrators: dict[str, Orchestrator] = {}
_cases: dict[str, CaseFile] = {}   # in-memory for the demo; use Cosmos DB or Postgres in production


def orch(pack: str) -> Orchestrator:
    if pack not in _orchestrators:
        _orchestrators[pack] = Orchestrator(get_pack(pack))
    return _orchestrators[pack]


class SubmitCase(BaseModel):
    case_id: str
    documents: dict[str, str]
    pack: str = "pnc_underwriting"


class Approval(BaseModel):
    reviewer: str
    decision: Decision
    note: str = ""


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/cases")
def submit(req: SubmitCase) -> dict:
    case = orch(req.pack).run(req.case_id, req.documents)
    _cases[case.case_id] = case
    return {"case_id": case.case_id, "recommendation": case.recommendation,
            "risk_score": case.risk_score, "memo_markdown": case.memo_markdown,
            "findings": [f.model_dump() for f in case.findings], "timings_ms": case.timings_ms}


@app.post("/cases/{case_id}/decision")
def decide(case_id: str, body: Approval) -> dict:
    case = _cases.get(case_id)
    if not case:
        raise HTTPException(404, "case not found")
    orch(case.pack).approve(case, body.reviewer, body.decision, body.note)
    return {"case_id": case_id, "human_decision": case.human_decision,
            "agreed_with_copilot": case.human_decision == case.recommendation}
