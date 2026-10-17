"""P&C commercial underwriting domain pack."""
from __future__ import annotations

from pathlib import Path

from copilot.engine.pack import DomainPack
from copilot.engine.schema import CaseFile, Citation, Decision, Finding

from . import analysis, extract, rules, scoring
from .config import MODEL_REFER_PROB


class PnCUnderwritingPack(DomainPack):
    name = "pnc_underwriting"
    guideline_path = Path(__file__).with_name("guidelines.md")

    def extract(self, case: CaseFile) -> CaseFile:
        return extract.extract(case)

    def analyze(self, case: CaseFile) -> CaseFile:
        return analysis.analyze(case)

    def check_rules(self, case: CaseFile, retriever) -> CaseFile:
        self._retriever = retriever
        return rules.check_rules(case, retriever)

    def score(self, case: CaseFile) -> CaseFile:
        return scoring.score(case)

    def decide(self, case: CaseFile) -> CaseFile:
        rule_findings = [f for f in case.findings if f.step == "rules"]
        if any(f.severity == "block" for f in rule_findings):
            case.recommendation = Decision.DECLINE
        elif rule_findings:
            case.recommendation = Decision.REFER
        elif case.risk_score is not None and case.risk_score >= MODEL_REFER_PROB:
            case.recommendation = Decision.REFER
            chunk = self._retriever.search("modeled probability large loss model referral", k=1)[0][0]
            case.findings.append(Finding(
                step="decide", severity="warn",
                message=f"[R13] Modeled large-loss probability {case.risk_score:.0%} is at or above "
                        f"{MODEL_REFER_PROB:.0%}; referred for senior review.",
                citations=[Citation(source=f"{chunk.source}#{chunk.section_id}", locator=chunk.section_id,
                                    snippet=chunk.text.replace(chr(10), ' ')[:160])]))
        else:
            case.recommendation = Decision.ACCEPT
        return case

    def draft_memo(self, case: CaseFile) -> str:
        v = case.value
        m = case.metrics
        rec = case.recommendation.value.upper() if case.recommendation else "N/A"

        def cite(f: Finding) -> str:
            return " ".join(f"[{c.source}{' ' + c.locator if c.locator and '#' not in c.source else ''}]"
                            for c in f.citations)

        def money(x):
            return f"${x:,.0f}" if isinstance(x, (int, float)) else "not provided"

        def flag(name):
            f = case.fields.get(name)
            if f is not None and f.value is not None and f.confidence < 0.8:
                return f" (UNVERIFIED: \"{f.citation.snippet}\", confirm with broker)" if f.citation else " (UNVERIFIED)"
            return ""

        lines = [
            f"# Underwriting Memo: {v('applicant_name', 'Unknown applicant')}",
            f"**Case:** {case.case_id} | **Copilot recommendation:** {rec} | "
            f"**Status:** awaiting underwriter approval",
            "",
            "## Account summary",
            f"- State: {v('state', 'not provided')} | NAICS: {v('naics', 'not provided')}{flag('naics')} | "
            f"Years in business: {v('years_in_business', 'not provided')}",
            f"- Annual revenue: {money(v('annual_revenue'))}{flag('annual_revenue')} | TIV: {money(v('tiv'))} "
            f"across {v('locations', '?')} location(s)",
            f"- Protection class (worst): {v('protection_class', 'n/a')} | "
            f"Fully sprinklered: {'yes' if v('all_sprinklered') else 'no'} | "
            f"Oldest building: {v('oldest_year_built', 'n/a')}",
            "",
            "## Loss history",
        ]
        if "loss_ratio_5y" in m:
            lines.append(f"- Loss ratio: {m['loss_ratio_5y']:.0%} | Claims: {int(m['claim_count_5y'])} | "
                         f"Largest loss: {money(m['largest_loss'])}")
        for f in case.findings:
            if f.step == "analysis":
                lines.append(f"- {f.message} {cite(f)}")

        lines += ["", "## Guideline checks"]
        rule_f = [f for f in case.findings if f.step in ("rules", "decide")]
        if rule_f:
            for f in rule_f:
                icon = "DECLINE" if f.severity == "block" else "REFER"
                lines.append(f"- **{icon}:** {f.message} {cite(f)}")
        else:
            lines.append("- All guideline checks passed.")

        lines += ["", "## Modeled risk"]
        if case.risk_score is not None:
            lines.append(f"- Large-loss probability: {case.risk_score:.0%} (advisory; refer threshold "
                         f"{MODEL_REFER_PROB:.0%})")
            lines.append("- Drivers, relative to the portfolio average:")
            for d in case.risk_drivers:
                direction = "raises" if d["contribution"] > 0 else "lowers"
                lines.append(f"  - {d['feature']} {direction} risk ({d['contribution']:+.2f})")
        else:
            lines.append("- Not scored: required inputs missing.")

        lines += ["", "## Underwriter action",
                  "- [ ] Approve recommendation  - [ ] Override (document reason)",
                  "", "_Generated by Risk Decision Copilot. Advisory only; a licensed underwriter "
                  "makes the final decision._"]
        return "\n".join(lines)
