"""Commercial lending domain pack (scaffold).

Same engine, different industry: financial statements in, credit-policy checks
out, credit memo for a loan officer. Implementing it is the fastest way to show
the engine is reusable. Suggested mapping from the P&C pack:

    P&C underwriting            ->  Commercial lending
    application.txt (ACORD)     ->  loan_application.txt
    sov.txt (property values)   ->  financials.txt (income statement, balance sheet)
    loss_run.txt                ->  credit_report.txt (payment history, delinquencies)
    loss ratio                  ->  DSCR = net operating income / annual debt service
    TIV limit                   ->  loan-to-value and maximum exposure per borrower
    appetite classes            ->  industry concentration limits
    large-loss model            ->  probability-of-default (PD) model

Policy thresholds to encode (fictional): DSCR < 1.25 refer, < 1.0 decline;
LTV > 80% refer; leverage (debt/EBITDA) > 4x refer; any 90+ day delinquency refer.
"""
from __future__ import annotations

from pathlib import Path

from copilot.engine.pack import DomainPack
from copilot.engine.schema import CaseFile


class CommercialLendingPack(DomainPack):
    name = "commercial_lending"
    guideline_path = Path(__file__).with_name("credit_policy.md")

    def _todo(self, case: CaseFile) -> CaseFile:
        raise NotImplementedError("Lending pack is a scaffold. See module docstring for the mapping.")

    extract = analyze = score = decide = _todo

    def check_rules(self, case: CaseFile, retriever) -> CaseFile:
        return self._todo(case)

    def draft_memo(self, case: CaseFile) -> str:
        return "Not implemented"
