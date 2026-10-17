"""Appetite agent: apply guideline rules and ground each outcome in retrieved text.

Each rule is deterministic (so the decision is auditable), and each triggered rule
retrieves the guideline passage that justifies it (so the memo cites policy, not
model memory). `expected_section` lets the eval harness score retrieval quality.
"""
from __future__ import annotations

from dataclasses import dataclass

from copilot.engine.schema import CaseFile, Citation, ExtractedField, Finding

from . import config as C


@dataclass
class RuleHit:
    rule_id: str
    outcome: str            # "refer" | "decline"
    message: str
    query: str              # what we ask the retriever
    expected_section: str   # ground truth for retrieval evals
    evidence: Citation | None = None


def evaluate_rules(case: CaseFile) -> list[RuleHit]:
    hits: list[RuleHit] = []
    v = case.value
    ev = lambda name: case.fields[name].citation if name in case.fields else None  # noqa: E731

    naics = v("naics")
    if naics in C.PROHIBITED_CLASSES:
        hits.append(RuleHit("R1", "decline", f"NAICS {naics} ({C.PROHIBITED_CLASSES[naics][0]}) is a prohibited class.",
                            "prohibited classes do not quote decline explosives petroleum", "2.3", ev("naics")))
    elif naics in C.REFERRAL_CLASSES:
        hits.append(RuleHit("R2", "refer", f"NAICS {naics} ({C.REFERRAL_CLASSES[naics][0]}) is a referral class.",
                            "referral classes elevated severity supplemental hazard questionnaire", "2.2", ev("naics")))
    elif naics and naics not in C.TARGET_CLASSES:
        hits.append(RuleHit("R3", "refer", f"NAICS {naics} is not listed in the appetite guide.",
                            "NAICS code not listed outside documented appetite class determination", "2.4", ev("naics")))

    tiv = v("tiv")
    if tiv is not None and tiv > C.TIV_DECLINE:
        hits.append(RuleHit("R4", "decline", f"TIV ${tiv:,.0f} exceeds the ${C.TIV_DECLINE:,.0f} account maximum.",
                            "maximum total insured value TIV per account", "3.1", ev("tiv")))
    elif tiv is not None and tiv > C.TIV_REFER:
        hits.append(RuleHit("R4", "refer", f"TIV ${tiv:,.0f} is above the ${C.TIV_REFER:,.0f} referral threshold.",
                            "TIV above referral capacity reinsurance review", "3.1", ev("tiv")))

    pc, spr = v("protection_class"), v("all_sprinklered")
    if pc is not None and pc >= 9:
        outcome = "refer" if spr else "decline"
        hits.append(RuleHit("R5", outcome,
                            f"Protection class {pc}; {'all buildings sprinklered' if spr else 'not fully sprinklered'}.",
                            "protection class 9 or 10 fire department sprinklered", "3.2", ev("protection_class")))

    oldest = v("oldest_year_built")
    if oldest is not None and oldest < C.OLD_BUILDING_YEAR:
        hits.append(RuleHit("R6", "refer", f"Oldest building constructed in {oldest} (before {C.OLD_BUILDING_YEAR}).",
                            "buildings constructed before 1960 updated roof electrical plumbing", "3.3",
                            ev("oldest_year_built")))

    lr = case.metrics.get("loss_ratio_5y")
    if lr is not None and lr > C.LR_DECLINE:
        hits.append(RuleHit("R7", "decline", f"Five-year loss ratio {lr:.0%} exceeds {C.LR_DECLINE:.0%}.",
                            "five-year loss ratio above 100% decline", "4.1", ev("loss_years")))
    elif lr is not None and lr > C.LR_REFER:
        hits.append(RuleHit("R7", "refer", f"Five-year loss ratio {lr:.0%} exceeds {C.LR_REFER:.0%}.",
                            "loss ratio thresholds above 65% referral", "4.1", ev("loss_years")))

    largest = case.metrics.get("largest_loss")
    if largest is not None and largest >= C.LARGE_LOSS:
        hits.append(RuleHit("R8", "refer", f"Single loss of ${largest:,.0f} is at or above ${C.LARGE_LOSS:,.0f}.",
                            "single claim incurred 250,000 large severe loss", "4.2", ev("loss_years")))

    years = v("loss_years", 0)
    if years < C.MIN_LOSS_YEARS:
        msg = "Loss runs missing." if years == 0 else f"Only {years} year(s) of loss history provided."
        hits.append(RuleHit("R9", "refer", msg,
                            "loss runs missing fewer than three years of history", "4.3", ev("loss_years")))

    yib = v("years_in_business")
    if yib is not None and yib < C.MIN_YEARS_IN_BUSINESS:
        hits.append(RuleHit("R10", "refer", f"{yib} year(s) in business (minimum {C.MIN_YEARS_IN_BUSINESS}).",
                            "businesses operating fewer than three years track record", "5.1",
                            ev("years_in_business")))

    state = v("state")
    if state and state not in C.LICENSED_STATES:
        hits.append(RuleHit("R11", "decline", f"Domiciled in {state}, where the program is not licensed.",
                            "licensed states risks domiciled in other states", "5.2", ev("state")))

    missing = [f for f in REQUIRED_FIELDS
               if f not in case.fields or case.fields[f].value is None
               or case.fields[f].confidence < MIN_CONFIDENCE]
    if missing:
        hits.append(RuleHit("R12", "refer", f"Missing or low-confidence required information: {', '.join(missing)}.",
                            "required information incomplete submissions returned to broker", "6.1",
                            next((case.fields[f].citation for f in missing if f in case.fields), None)))
    return hits


REQUIRED_FIELDS = ["applicant_name", "state", "naics", "years_in_business", "annual_revenue", "tiv"]
MIN_CONFIDENCE = 0.8


def check_rules(case: CaseFile, retriever) -> CaseFile:
    hits = evaluate_rules(case)
    for hit in hits:
        top = retriever.search(hit.query, k=3)
        best_chunk = top[0][0]
        cites = [Citation(source=f"{best_chunk.source}#{best_chunk.section_id}",
                          locator=best_chunk.section_id,
                          snippet=best_chunk.text.replace("\n", " ")[:160])]
        if hit.evidence:
            cites.append(hit.evidence)
        case.findings.append(Finding(
            step="rules", severity="block" if hit.outcome == "decline" else "warn",
            message=f"[{hit.rule_id}] {hit.message}", citations=cites))
        # Retrieval telemetry for the eval harness.
        case.fields.setdefault("_retrieval_log", ExtractedField(name="_retrieval_log", value=[]))
        case.fields["_retrieval_log"].value.append({
            "rule": hit.rule_id, "outcome": hit.outcome, "expected": hit.expected_section,
            "top3": [c.section_id for c, _ in top],
        })
    return case
