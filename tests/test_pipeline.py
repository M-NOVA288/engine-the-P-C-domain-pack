from pathlib import Path

import pytest

from copilot.engine import CaseFile, Decision, Orchestrator
from copilot.packs.pnc.extract import extract, parse_money
from copilot.packs.pnc.pack import PnCUnderwritingPack
from roi.roi import Assumptions, compute

ROOT = Path(__file__).resolve().parents[1]

APP = """Named Insured: Test Bakery LLC
Mailing State: TX
NAICS Code: 311811
Years in Business: 12
Annual Revenue: $2.50 million
"""
SOV = """STATEMENT OF VALUES
Loc | Address | Building Value | Contents Value | BI Value | Construction | Year Built | Sprinklered | Protection Class
----
1 | 1 Main St | 1,000,000 | 200,000 | 100K | Masonry NC | 1995 | Yes | PC 3
2 | 2 Main St | $500,000 | 100,000 | 50,000 | Frame | 2005 | Y | 4
TOTAL | | 1,500,000 | 300,000 | 150,000 | | | |
"""
LOSS = """Policy Year | Earned Premium | Claims | Paid | Reserves | Incurred
----
2021-2022 | 10,000 | 0 | 0 | 0 | 0
2022-2023 | 10,000 | 1 | 2,000 | 0 | 2,000
2023-2024 | 10,000 | 0 | 0 | 0 | 0
2024-2025 | 10,000 | 1 | 3,000 | 0 | 3,000
2025-2026 | 10,000 | 0 | 0 | 0 | 0

Claim Detail
Date of Loss | Cause | Incurred | Status
2022-05-01 | Theft | 2,000 | Closed
2024-05-01 | Theft | 3,000 | Closed
"""


@pytest.mark.parametrize("text,expected", [
    ("$4,250,000", 4_250_000), ("4.25M", 4_250_000), ("$4.25 million", 4_250_000),
    ("4,250K", 4_250_000), ("12,000", 12_000),
])
def test_parse_money(text, expected):
    assert parse_money(text) == expected


def test_extract_skips_total_row():
    case = extract(CaseFile(case_id="t", pack="p", documents={"application.txt": APP, "sov.txt": SOV}))
    assert case.value("tiv") == 1_950_000
    assert case.value("locations") == 2
    assert case.value("protection_class") == 4
    assert case.value("all_sprinklered") is True


def test_ocr_damaged_naics_is_low_confidence():
    case = extract(CaseFile(case_id="t", pack="p",
                            documents={"application.txt": APP.replace("311811", "3ll8ll")}))
    assert case.value("naics") == "311811"
    assert case.fields["naics"].confidence < 0.8


def _run(app=APP, sov=SOV, loss=LOSS, tmp_path=None):
    orch = Orchestrator(PnCUnderwritingPack(), audit_path=tmp_path / "audit.jsonl")
    docs = {"application.txt": app, "sov.txt": sov}
    if loss:
        docs["loss_run.txt"] = loss
    return orch, orch.run("T-1", docs)


def test_clean_account_accepted(tmp_path):
    _, case = _run(tmp_path=tmp_path)
    assert case.metrics["loss_ratio_5y"] == 0.1
    assert case.recommendation == Decision.ACCEPT


def test_missing_loss_runs_referred(tmp_path):
    _, case = _run(loss=None, tmp_path=tmp_path)
    assert case.recommendation == Decision.REFER
    assert any("[R9]" in f.message for f in case.findings)


def test_unlicensed_state_declined_with_citation(tmp_path):
    _, case = _run(app=APP.replace("TX", "FL"), tmp_path=tmp_path)
    assert case.recommendation == Decision.DECLINE
    f = next(f for f in case.findings if "[R11]" in f.message)
    assert any(c.locator == "5.2" for c in f.citations)


def test_human_approval_is_audited(tmp_path):
    orch, case = _run(tmp_path=tmp_path)
    orch.approve(case, "reviewer-1", Decision.ACCEPT)
    log = (tmp_path / "audit.jsonl").read_text()
    assert "human_decision" in log and "reviewer-1" in log


def test_roi_math():
    out = compute(Assumptions(submissions_per_month=100, manual_minutes=60, assisted_minutes=10,
                              referral_minutes=30, referral_rate=0.5, salary=104_000,
                              benefits_load=0.0, build_and_run_cost_per_year=0))
    assert out["avg_assisted_minutes"] == 20
    assert out["hours_saved_per_month"] == pytest.approx(66.7, abs=0.1)
    assert out["gross_savings_per_year"] == round(66.6667 * 12 * 50)


def test_eval_gate_passes():
    import json
    from evals.run_evals import gate, run
    results = run()
    fails = gate(results, json.loads((ROOT / "evals" / "thresholds.json").read_text()))
    assert not fails, fails
