"""Loss analysis agent: loss ratio, frequency, severity and claim patterns."""
from __future__ import annotations

from collections import Counter

from copilot.engine.schema import CaseFile, Citation, Finding


def analyze(case: CaseFile) -> CaseFile:
    years = case.value("loss_run_years_detail", [])
    claims = case.value("loss_run_claims", [])
    if not years:
        case.findings.append(Finding(step="analysis", severity="warn",
                                     message="No loss run data available to analyze."))
        return case

    recent = years[-5:]
    premium = sum(y["earned_premium"] for y in recent)
    incurred = sum(y["incurred"] for y in recent)
    n_claims = sum(y["claims"] for y in recent)
    lr = incurred / premium if premium else 0.0

    case.metrics.update({
        "earned_premium_5y": premium,
        "incurred_5y": incurred,
        "loss_ratio_5y": round(lr, 4),
        "claim_count_5y": n_claims,
        "frequency_per_year": round(n_claims / len(recent), 2),
        "avg_severity": round(incurred / n_claims, 2) if n_claims else 0.0,
        "largest_loss": max((c["incurred"] for c in claims), default=0.0),
    })

    cite = Citation(source="loss_run.txt", locator=f"lines {recent[0]['line']}-{recent[-1]['line']}",
                    snippet=f"{len(recent)} policy years, EP ${premium:,.0f}, incurred ${incurred:,.0f}")
    case.findings.append(Finding(
        step="analysis", severity="info",
        message=(f"{len(recent)}-year loss ratio {lr:.0%} on ${premium:,.0f} earned premium; "
                 f"{n_claims} claims, average severity ${case.metrics['avg_severity']:,.0f}."),
        citations=[cite]))

    causes = Counter(c["cause"] for c in claims)
    if causes:
        top, count = causes.most_common(1)[0]
        if count >= 2:
            case.findings.append(Finding(
                step="analysis", severity="info",
                message=f"Recurring cause of loss: {top} ({count} of {len(claims)} claims).",
                citations=[Citation(source="loss_run.txt", locator="claim detail", snippet=top)]))

    # Trend: compare the latest two years with the earlier years.
    if len(recent) >= 4:
        def ratio(ys):
            p = sum(y["earned_premium"] for y in ys)
            return sum(y["incurred"] for y in ys) / p if p else 0.0
        early, late = ratio(recent[:-2]), ratio(recent[-2:])
        if late > early + 0.25:
            case.findings.append(Finding(
                step="analysis", severity="info",
                message=f"Deteriorating trend: last two years at {late:.0%} vs {early:.0%} prior.",
                citations=[cite]))
    return case
