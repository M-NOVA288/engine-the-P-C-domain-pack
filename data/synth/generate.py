"""Synthetic data generator.

Produces two things, all fictional (no real insureds or PII):

1. data/portfolio.csv: historical policies with a large-loss outcome, used to
   train the risk model.
2. data/submissions/<CASE>/: broker submissions (application, schedule of values,
   loss run) written in inconsistent real-world formats, plus labels.jsonl with
   the true field values and the decision a senior underwriter would make.

The labeling logic below is written independently of the copilot's rule engine,
so the evals test the copilot rather than the copilot grading itself.

    python -m data.synth.generate --cases 60 --policies 6000 --seed 7
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from copilot.packs.pnc import config as C

ROOT = Path(__file__).resolve().parents[2]
OUT_SUB = ROOT / "data" / "submissions"

FIRST = ["Ridgeview", "Lone Star", "Red River", "Prairie", "Summit", "Bluebonnet", "Ozark", "Mesa",
         "Cedar", "Ironwood", "Pecos", "Gulfway", "Sandia", "Canyon", "Brazos", "Timberline", "Arbor",
         "Granite", "Sabine", "Frontier", "Big Sky", "Cimarron", "Riverbend", "Copperline"]
SECOND = ["Metal Works", "Supply", "Holdings", "Bakery", "Mechanical", "Engineering", "Kitchen & Bar",
          "Properties", "Industries", "Fabrication", "Services", "Partners", "Group", "Logistics"]
SUFFIX = ["LLC", "Inc.", "Co.", "LP", "Corp."]
CAUSES = ["Fire", "Water damage", "Slip and fall", "Theft", "Wind/hail", "Products liability",
          "Equipment breakdown", "Vehicle impact"]
STREETS = ["Main St", "Industrial Blvd", "Commerce Dr", "Oak Ave", "Ranch Rd", "Market St", "Airport Way"]
CITIES = {"TX": "Austin", "OK": "Tulsa", "LA": "Baton Rouge", "AR": "Little Rock", "NM": "Albuquerque",
          "KS": "Wichita", "MO": "Springfield", "CO": "Pueblo", "FL": "Tampa", "GA": "Macon", "CA": "Fresno"}


# --------------------------------------------------------------------------- truth model
def true_probability(hazard, tiv, pc, sprinklered, age, claims, yib, revenue) -> float:
    """The 'real world' large-loss process. The copilot never sees this function;
    it has to learn an approximation from the portfolio data."""
    lin = (-3.3 + {"low": 0.0, "medium": 0.55, "high": 1.1}[hazard]
           + 0.14 * (pc - 4) - 0.75 * sprinklered + 0.012 * (age - 30)
           + (0.7 if age > 60 and not sprinklered else 0.0)          # interaction the model lacks
           + 0.30 * min(claims, 10) - 0.05 * min(yib, 25)
           + 0.35 * (math.log10(tiv) - 6.5) + 0.10 * (math.log10(revenue) - 6.5))
    return 1 / (1 + math.exp(-lin))


def round_to(x, step):
    return int(round(x / step) * step)


# --------------------------------------------------------------------------- formatting noise
def fmt_money(x: float, rng: random.Random) -> str:
    style = rng.choice(["comma", "dollar", "million", "k", "M"])
    if style == "comma":
        return f"{x:,.0f}"
    if style == "dollar":
        return f"${x:,.0f}"
    if style == "million" and x >= 1_000_000:
        return f"${x / 1e6:.2f} million"
    if style == "M" and x >= 1_000_000:
        return f"{x / 1e6:.2f}M"
    if style == "k" and x % 1000 == 0:
        return f"{x / 1000:,.0f}K"
    return f"${x:,.0f}"


def fmt_pc(pc, rng):
    return rng.choice([str(pc), f"PC {pc}", f"{pc:02d}", f"Class {pc}"])


def fmt_spr(s, rng):
    return rng.choice(["Y", "Yes", "100%"]) if s else rng.choice(["N", "No", "None"])


# --------------------------------------------------------------------------- one account
def sample_account(rng: random.Random) -> dict:
    r = rng.random()
    if r < 0.70:
        naics = rng.choice(list(C.TARGET_CLASSES))
    elif r < 0.82:
        naics = rng.choice(list(C.REFERRAL_CLASSES))
    elif r < 0.90:
        naics = rng.choice(list(C.PROHIBITED_CLASSES))
    else:
        naics = rng.choice(list(C.UNLISTED_EXAMPLES))
    hazard = C.hazard_of(naics)

    state = rng.choice(sorted(C.LICENSED_STATES)) if rng.random() < 0.9 else rng.choice(["FL", "GA", "CA"])
    yib = rng.randint(1, 2) if rng.random() < 0.08 else rng.randint(3, 45)

    n_loc = rng.choices([1, 2, 3, 4], weights=[45, 30, 15, 10])[0]
    scale = rng.choices([1.0, 4.0, 7.0], weights=[86, 9, 5])[0]
    locs = []
    for _ in range(n_loc):
        bldg = round_to(rng.uniform(300_000, 3_500_000) * scale, 10_000)
        locs.append({
            "building": bldg,
            "contents": round_to(bldg * rng.uniform(0.15, 0.6), 10_000),
            "bi": round_to(bldg * rng.uniform(0.05, 0.3), 10_000),
            "year": rng.randint(1925, 1959) if rng.random() < 0.12 else rng.randint(1960, 2022),
            "pc": rng.choice([9, 10]) if rng.random() < 0.07 else rng.choices(
                range(1, 9), weights=[3, 8, 12, 14, 12, 9, 6, 4])[0],
            "spr": rng.random() < 0.55,
            "construction": rng.choice(["Frame", "Joisted Masonry", "Non-combustible", "Masonry NC", "Fire Resistive"]),
        })
    tiv = sum(l["building"] + l["contents"] + l["bi"] for l in locs)
    revenue = round_to(max(tiv * rng.uniform(0.4, 2.5), 250_000), 10_000)
    return {"naics": naics, "hazard": hazard, "state": state, "yib": yib, "locs": locs,
            "tiv": tiv, "revenue": revenue,
            "pc": max(l["pc"] for l in locs), "all_spr": all(l["spr"] for l in locs),
            "oldest": min(l["year"] for l in locs),
            "employees": max(3, int(revenue / rng.uniform(90_000, 220_000)))}


def sample_losses(acct: dict, rng: random.Random, n_years: int) -> tuple[list, list]:
    base_premium = acct["tiv"] * 0.0035 + acct["revenue"] * 0.002
    freq = {"low": 0.25, "medium": 0.45, "high": 0.7}[acct["hazard"]] * rng.uniform(0.5, 1.8)
    years, claims = [], []
    for k in range(n_years):
        y0 = C.AS_OF_YEAR - n_years + k
        ep = round_to(base_premium * (0.92 + 0.03 * k) * rng.uniform(0.95, 1.05), 100)
        n = np.random.default_rng(rng.randint(0, 10**9)).poisson(freq)
        yr_claims = []
        for _ in range(n):
            if rng.random() < 0.03:
                inc = round_to(rng.uniform(250_000, 600_000), 500)
            else:
                inc = round_to(min(rng.lognormvariate(math.log(max(ep * 0.22, 2000)), 0.8), 240_000), 100)
            yr_claims.append({"date": f"{y0 + rng.randint(0, 1)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                              "cause": rng.choice(CAUSES), "incurred": inc,
                              "status": rng.choice(["Closed", "Closed", "Open"])})
        claims += yr_claims
        inc_total = sum(c["incurred"] for c in yr_claims)
        paid = round_to(inc_total * rng.uniform(0.6, 1.0), 100) if inc_total else 0
        years.append({"year": f"{y0}-{y0 + 1}", "ep": ep, "n": n, "paid": paid,
                      "reserve": inc_total - paid, "incurred": inc_total})
    return years, claims


# --------------------------------------------------------------------------- documents
def write_application(acct, name, omit_revenue, rng, quirks=()) -> str:
    L = ["COMMERCIAL INSURANCE APPLICATION (ACORD 125 FORMAT - SYNTHETIC)",
         f"Agency: {rng.choice(['Southwest Risk Partners', 'Heartland Brokerage', 'Tri-State Insurance Group'])}",
         f"Producer Code: {rng.randint(10000, 99999)}", ""]
    L.append(f"{rng.choice(['Applicant Name', 'Named Insured'])}: {name}")
    city = CITIES[acct["state"]]
    L.append(f"Mailing Address: {rng.randint(100, 9999)} {rng.choice(STREETS)}, {city}")
    L.append(f"{rng.choice(['Mailing State', 'State of Domicile'])}: {acct['state']}")
    naics_txt = acct["naics"]
    if "ocr" in quirks:  # scanned form: OCR reads 1 as l and 0 as O
        naics_txt = naics_txt.replace("1", "l").replace("0", "O")
    L.append(f"{rng.choice(['NAICS Code', 'Primary NAICS'])}: {naics_txt}")
    if rng.random() < 0.5:
        L.append(f"Years in Business: {acct['yib']}")
    else:
        L.append(f"{rng.choice(['Year Established', 'Date Business Started'])}: {C.AS_OF_YEAR - acct['yib']}")
    if not omit_revenue and "revenue_in_remarks" not in quirks:
        L.append(f"{rng.choice(['Annual Revenue', 'Gross Sales', 'Revenues'])}: {fmt_money(acct['revenue'], rng)}")
    L.append(f"{rng.choice(['Employees', 'Number of Employees'])}: {acct['employees']}")
    L.append("Lines Requested: Commercial Property | General Liability")
    L.append(f"Requested Effective Date: {C.AS_OF_YEAR + 1}-01-01")
    L.append("")
    remark = rng.choice(["Renewal marketing, incumbent non-renewing.",
                         "New venture expansion, see attached SOV.",
                         "Shopping for better terms; clean account per broker."])
    if "revenue_in_remarks" in quirks:  # broker put sales in free text instead of the field
        remark += f" Insured reports sales of roughly {fmt_money(acct['revenue'], rng)} last fiscal year."
    L.append("Remarks: " + remark)
    return "\n".join(L) + "\n"


def write_sov(acct, rng, quirks=()) -> str:
    hdr = rng.choice([
        "Loc | Address | Building Value | Contents Value | BI Value | Construction | Year Built | Sprinklered | Protection Class",
        "Loc # | Location | Building | BPP | Business Income | Const | Yr Built | Sprinkler | PC",
    ])
    L = ["STATEMENT OF VALUES", hdr, "-" * 40]
    for i, l in enumerate(acct["locs"], start=1):
        L.append(" | ".join([str(i), f"{rng.randint(100, 9999)} {rng.choice(STREETS)}",
                             fmt_money(l["building"], rng), fmt_money(l["contents"], rng),
                             fmt_money(l["bi"], rng), l["construction"], str(l["year"]),
                             fmt_spr(l["spr"], rng), fmt_pc(l["pc"], rng)]))
    if "sov_total" in quirks:  # many SOVs end with a totals row that must not be summed again
        tb = sum(l["building"] for l in acct["locs"]); tc = sum(l["contents"] for l in acct["locs"])
        tbi = sum(l["bi"] for l in acct["locs"])
        L.append(" | ".join(["TOTAL", "", f"{tb:,}", f"{tc:,}", f"{tbi:,}", "", "", "", ""]))
    return "\n".join(L) + "\n"


def write_loss_run(years, claims, rng) -> str:
    L = [f"LOSS RUN REPORT - Carrier: {rng.choice(['Synthetic Mutual', 'Example Casualty Co.', 'Demo Indemnity'])}",
         f"Valued as of: {C.AS_OF_YEAR}-09-30", "",
         "Policy Year | Earned Premium | Claims | Paid | Reserves | Incurred", "-" * 40]
    for y in years:
        L.append(f"{y['year']} | {fmt_money(y['ep'], rng)} | {y['n']} | {y['paid']:,} | {y['reserve']:,} | {y['incurred']:,}")
    L += ["", "Claim Detail", "Date of Loss | Cause | Incurred | Status", "-" * 40]
    for c in claims:
        L.append(f"{c['date']} | {c['cause']} | {c['incurred']:,} | {c['status']}")
    if not claims:
        L.append("No claims reported.")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- labeling
def senior_underwriter_label(acct, years, claims, omit_revenue) -> tuple[str, list[str], dict]:
    """Independent implementation of the guideline decision on TRUE values."""
    declines, refers = [], []
    n = acct["naics"]
    if n in C.PROHIBITED_CLASSES:
        declines.append("2.3")
    elif n in C.REFERRAL_CLASSES:
        refers.append("2.2")
    elif n not in C.TARGET_CLASSES:
        refers.append("2.4")
    if acct["tiv"] > 25_000_000:
        declines.append("3.1")
    elif acct["tiv"] > 15_000_000:
        refers.append("3.1")
    if acct["pc"] >= 9:
        (refers if acct["all_spr"] else declines).append("3.2")
    if acct["oldest"] < 1960:
        refers.append("3.3")

    ep = sum(y["ep"] for y in years[-5:])
    inc = sum(y["incurred"] for y in years[-5:])
    lr = inc / ep if ep else None
    largest = max((c["incurred"] for c in claims), default=0)
    if lr is not None and lr > 1.0:
        declines.append("4.1")
    elif lr is not None and lr > 0.65:
        refers.append("4.1")
    if largest >= 250_000:
        refers.append("4.2")
    if len(years) < 3:
        refers.append("4.3")
    if acct["yib"] < 3:
        refers.append("5.1")
    if acct["state"] not in C.LICENSED_STATES:
        declines.append("5.2")
    if omit_revenue:
        refers.append("6.1")

    n_claims = sum(y["n"] for y in years[-5:])
    p = true_probability(acct["hazard"], acct["tiv"], acct["pc"], acct["all_spr"],
                         C.AS_OF_YEAR - acct["oldest"], n_claims, acct["yib"], acct["revenue"])
    if declines:
        decision = "decline"
    elif refers:
        decision = "refer"
    elif p >= 0.25:
        decision, refers = "refer", ["7.1"]
    else:
        decision = "accept"
    stats = {"loss_ratio_5y": round(lr, 4) if lr is not None else None, "claim_count_5y": n_claims,
             "largest_loss": largest, "true_prob": round(p, 4)}
    return decision, sorted(set(declines + refers)), stats


# --------------------------------------------------------------------------- main
def generate_cases(n_cases: int, seed: int) -> None:
    rng = random.Random(seed)
    if OUT_SUB.exists():
        shutil.rmtree(OUT_SUB)
    OUT_SUB.mkdir(parents=True)
    used, labels = set(), []
    for i in range(1, n_cases + 1):
        acct = sample_account(rng)
        while True:
            name = f"{rng.choice(FIRST)} {rng.choice(SECOND)} {rng.choice(SUFFIX)}"
            if name not in used:
                used.add(name)
                break
        r = rng.random()
        n_years = 0 if r < 0.08 else (2 if r < 0.15 else (4 if r < 0.20 else 5))
        years, claims = sample_losses(acct, rng, n_years)
        omit_revenue = rng.random() < 0.06
        quirks = []
        if any(ch in acct["naics"] for ch in "01") and rng.random() < 0.10:
            quirks.append("ocr")
        if not omit_revenue and rng.random() < 0.08:
            quirks.append("revenue_in_remarks")
        if rng.random() < 0.30:
            quirks.append("sov_total")

        case_id = f"CASE-{i:04d}"
        d = OUT_SUB / case_id
        d.mkdir()
        (d / "application.txt").write_text(write_application(acct, name, omit_revenue, rng, quirks))
        (d / "sov.txt").write_text(write_sov(acct, rng, quirks))
        if n_years:
            (d / "loss_run.txt").write_text(write_loss_run(years, claims, rng))

        decision, sections, stats = senior_underwriter_label(acct, years, claims, omit_revenue)
        labels.append({
            "case_id": case_id, "decision": decision, "sections": sections, "quirks": quirks,
            "truth": {
                "applicant_name": name, "state": acct["state"], "naics": acct["naics"],
                "years_in_business": acct["yib"], "annual_revenue": None if omit_revenue else acct["revenue"],
                "tiv": acct["tiv"], "protection_class": acct["pc"], "all_sprinklered": acct["all_spr"],
                "oldest_year_built": acct["oldest"], "loss_years": n_years,
                **{k: v for k, v in stats.items() if k != "true_prob"},
            },
            "true_prob": stats["true_prob"],
        })
    with (OUT_SUB / "labels.jsonl").open("w") as f:
        for row in labels:
            f.write(json.dumps(row) + "\n")
    mix = pd.Series([l["decision"] for l in labels]).value_counts().to_dict()
    print(f"Wrote {n_cases} submissions to {OUT_SUB.relative_to(ROOT)}  decision mix: {mix}")


def generate_portfolio(n: int, seed: int) -> None:
    rng = random.Random(seed + 1)
    rows = []
    for _ in range(n):
        acct = sample_account(rng)
        years, _ = sample_losses(acct, rng, 5)
        claims = sum(y["n"] for y in years)
        p = true_probability(acct["hazard"], acct["tiv"], acct["pc"], acct["all_spr"],
                             C.AS_OF_YEAR - acct["oldest"], claims, acct["yib"], acct["revenue"])
        rows.append({"naics": acct["naics"], "tiv": acct["tiv"], "protection_class": acct["pc"],
                     "all_sprinklered": acct["all_spr"], "oldest_year_built": acct["oldest"],
                     "claim_count_5y": claims, "years_in_business": acct["yib"],
                     "annual_revenue": acct["revenue"], "large_loss_next_year": int(rng.random() < p)})
    df = pd.DataFrame(rows)
    out = ROOT / "data" / "portfolio.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} policies to {out.relative_to(ROOT)}  large-loss rate: "
          f"{df.large_loss_next_year.mean():.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=60)
    ap.add_argument("--policies", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    generate_portfolio(a.policies, a.seed)
    generate_cases(a.cases, a.seed)
