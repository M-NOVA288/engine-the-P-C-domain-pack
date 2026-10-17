"""ROI model for the copilot. Every assumption is an explicit, editable input.

    python -m roi.roi --underwriters 8 --submissions-per-month 600 --salary 110000

The referral rate defaults to the rate measured by the eval harness, which ties
the business case to the system's actual behavior rather than a guess.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def measured_referral_rate(default: float = 0.35) -> float:
    p = ROOT / "evals" / "results.json"
    return json.loads(p.read_text())["referral_rate"] if p.exists() else default


@dataclass
class Assumptions:
    underwriters: int = 8
    submissions_per_month: int = 600
    salary: float = 110_000            # example base salary; replace with your team's
    benefits_load: float = 0.30        # benefits and overhead on top of salary
    hours_per_year: float = 2080
    manual_minutes: float = 45         # time to triage and write a memo by hand (measure this)
    assisted_minutes: float = 12       # review time when the copilot's memo is accept/decline
    referral_minutes: float = 30       # review time when the copilot refers the case
    referral_rate: float = 0.35
    build_and_run_cost_per_year: float = 60_000  # cloud, model usage, maintenance (estimate)


def compute(a: Assumptions) -> dict:
    hourly = a.salary * (1 + a.benefits_load) / a.hours_per_year
    assisted_avg = a.referral_rate * a.referral_minutes + (1 - a.referral_rate) * a.assisted_minutes
    minutes_saved_each = max(a.manual_minutes - assisted_avg, 0)
    hours_saved_month = a.submissions_per_month * minutes_saved_each / 60
    gross_savings_year = hours_saved_month * 12 * hourly
    net_savings_year = gross_savings_year - a.build_and_run_cost_per_year

    workdays_month = 21
    uw_hours_day = 6.5  # productive hours
    manual_capacity = uw_hours_day * 60 / a.manual_minutes
    assisted_capacity = uw_hours_day * 60 / assisted_avg
    return {
        "loaded_hourly_cost": round(hourly, 2),
        "avg_assisted_minutes": round(assisted_avg, 1),
        "minutes_saved_per_submission": round(minutes_saved_each, 1),
        "hours_saved_per_month": round(hours_saved_month, 1),
        "fte_equivalent_freed": round(hours_saved_month / (uw_hours_day * workdays_month), 2),
        "gross_savings_per_year": round(gross_savings_year),
        "net_savings_per_year": round(net_savings_year),
        "roi_multiple": round(gross_savings_year / a.build_and_run_cost_per_year, 2)
        if a.build_and_run_cost_per_year else None,
        "submissions_per_uw_per_day_manual": round(manual_capacity, 1),
        "submissions_per_uw_per_day_assisted": round(assisted_capacity, 1),
        "capacity_uplift_pct": round(100 * (assisted_capacity / manual_capacity - 1), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    defaults = Assumptions(referral_rate=measured_referral_rate())
    for k, v in asdict(defaults).items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = vars(ap.parse_args())
    a = Assumptions(**args)
    out = compute(a)
    print("Assumptions:", json.dumps(asdict(a), indent=2))
    print("Results:", json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
