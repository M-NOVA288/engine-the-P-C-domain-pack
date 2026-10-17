"""Golden-set evaluation and release gate.

    python -m evals.run_evals            # prints report, writes evals/results.json + evals/report.md
    python -m evals.run_evals --gate     # also exits non-zero if thresholds fail (used in CI)

Metrics are chosen to answer what a business owner asks, not just what an ML
engineer asks:
  * Does it make the same call a senior underwriter would?   decision accuracy
  * Does it ever wave through a risk it should not?          unsafe accepts (must be 0)
  * Does it ever wrongly decline good business?              wrongful declines (must be 0)
  * Can I trust the numbers it pulled from the documents?    field accuracy
  * Does it cite the right guideline?                        retrieval hit@1 / hit@3
  * Can every warning be traced to a source?                 citation coverage
  * Same input, same answer?                                 consistency
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from copilot.engine import Orchestrator
from copilot.packs.pnc.pack import PnCUnderwritingPack

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "data" / "submissions"

NUMERIC_TOL = {"annual_revenue": 0.005, "tiv": 0.005}   # relative tolerance
FIELDS = ["applicant_name", "state", "naics", "years_in_business", "annual_revenue", "tiv",
          "protection_class", "all_sprinklered", "oldest_year_built", "loss_years"]


def field_correct(name, got, truth) -> bool:
    if truth is None:               # field intentionally absent from the documents
        return got is None
    if got is None:
        return False
    if name in NUMERIC_TOL:
        return abs(got - truth) <= NUMERIC_TOL[name] * truth
    return got == truth


def run() -> dict:
    labels = [json.loads(l) for l in (SUB / "labels.jsonl").read_text().splitlines()]
    audit = Path(tempfile.mkdtemp()) / "eval_audit.jsonl"
    orch = Orchestrator(PnCUnderwritingPack(), audit_path=audit)

    confusion = Counter()
    field_hits, field_total = defaultdict(int), defaultdict(int)
    retr_1 = retr_3 = retr_n = 0
    cited = warn_total = 0
    consistent = 0
    latencies, errors, per_case = [], [], []
    lr_abs_err = []

    for lab in labels:
        d = SUB / lab["case_id"]
        docs = {f.name: f.read_text() for f in sorted(d.iterdir())}
        case = orch.run(lab["case_id"], docs)
        rerun = orch.run(lab["case_id"], docs)
        consistent += hashlib.sha256(case.memo_markdown.encode()).digest() == \
            hashlib.sha256(rerun.memo_markdown.encode()).digest()

        got, truth = case.recommendation.value, lab["decision"]
        confusion[(truth, got)] += 1
        latencies.append(case.timings_ms["total"])

        for f in FIELDS:
            val = case.value(f, 0 if f == "loss_years" else None)
            # Low-confidence values are surfaced for review, not trusted; score them as not extracted.
            if f in case.fields and case.fields[f].confidence < 0.8:
                val = None if lab["truth"][f] is not None else val
            field_total[f] += 1
            field_hits[f] += field_correct(f, val, lab["truth"][f])

        if lab["truth"]["loss_ratio_5y"] is not None and "loss_ratio_5y" in case.metrics:
            lr_abs_err.append(abs(case.metrics["loss_ratio_5y"] - lab["truth"]["loss_ratio_5y"]))

        for r in case.value("_retrieval_log", []):
            retr_n += 1
            retr_1 += r["top3"][0] == r["expected"]
            retr_3 += r["expected"] in r["top3"]

        for f in case.findings:
            if f.severity in ("warn", "block") and f.step in ("rules", "decide"):
                warn_total += 1
                cited += bool(f.citations)

        if got != truth:
            kind = ("UNSAFE ACCEPT" if got == "accept" else
                    "WRONGFUL DECLINE" if got == "decline" else "over-referral (safe)")
            errors.append({"case_id": lab["case_id"], "truth": truth, "copilot": got, "kind": kind,
                           "quirks": lab.get("quirks", [])})
        per_case.append({"case_id": lab["case_id"], "truth": truth, "copilot": got,
                         "risk_score": case.risk_score, "quirks": lab.get("quirks", [])})

    n = len(labels)
    correct = sum(v for (t, g), v in confusion.items() if t == g)
    should_not_accept = sum(v for (t, _), v in confusion.items() if t != "accept")
    caught = sum(v for (t, g), v in confusion.items() if t != "accept" and g != "accept")
    quirky = [c for c in per_case if c["quirks"] and set(c["quirks"]) != {"sov_total"}]

    results = {
        "n_cases": n,
        "decision_accuracy": round(correct / n, 4),
        "unsafe_accepts": sum(v for (t, g), v in confusion.items() if g == "accept" and t != "accept"),
        "wrongful_declines": sum(v for (t, g), v in confusion.items() if g == "decline" and t != "decline"),
        "over_referrals": sum(v for (t, g), v in confusion.items() if g == "refer" and t == "accept"),
        "risk_capture_rate": round(caught / should_not_accept, 4) if should_not_accept else 1.0,
        "field_accuracy": round(sum(field_hits.values()) / sum(field_total.values()), 4),
        "field_accuracy_by_field": {f: round(field_hits[f] / field_total[f], 4) for f in FIELDS},
        "loss_ratio_mean_abs_error_pts": round(100 * statistics.mean(lr_abs_err), 3) if lr_abs_err else None,
        "retrieval_hit_at_1": round(retr_1 / retr_n, 4) if retr_n else None,
        "retrieval_hit_at_3": round(retr_3 / retr_n, 4) if retr_n else None,
        "citation_coverage": round(cited / warn_total, 4) if warn_total else 1.0,
        "consistency": round(consistent / n, 4),
        "latency_ms_median": round(statistics.median(latencies), 1),
        "latency_ms_p95": round(sorted(latencies)[int(0.95 * (n - 1))], 1),
        "messy_document_cases": len(quirky),
        "messy_cases_handled_safely": sum(c["copilot"] != "accept" or c["truth"] == "accept" for c in quirky),
        "decision_mix_truth": dict(Counter(l["decision"] for l in labels)),
        "decision_mix_copilot": dict(Counter(c["copilot"] for c in per_case)),
        "referral_rate": round(sum(c["copilot"] == "refer" for c in per_case) / n, 4),
        "confusion": {f"{t}->{g}": v for (t, g), v in sorted(confusion.items())},
        "errors": errors,
    }
    results["retrieval_qa"] = retrieval_qa()
    return results


def retrieval_qa() -> dict:
    """Harder retrieval test: paraphrased questions an underwriter would actually type."""
    from copilot.engine.retrieval import get_retriever
    pack = PnCUnderwritingPack()
    retriever = get_retriever(pack.guideline_path)
    qs = json.loads((ROOT / "evals" / "retrieval_questions.json").read_text())
    h1 = h3 = 0
    misses = []
    for q in qs:
        top = [c.section_id for c, _ in retriever.search(q["question"], k=3)]
        h1 += top[0] == q["expected"]
        h3 += q["expected"] in top
        if top[0] != q["expected"]:
            misses.append({"question": q["question"], "expected": q["expected"], "got_top3": top})
    return {"n": len(qs), "hit_at_1": round(h1 / len(qs), 4), "hit_at_3": round(h3 / len(qs), 4),
            "misses": misses}


def gate(results: dict, th: dict) -> list[str]:
    fails = []
    checks = [
        ("decision_accuracy", ">=", th["decision_accuracy_min"]),
        ("unsafe_accepts", "<=", th["unsafe_accepts_max"]),
        ("wrongful_declines", "<=", th["wrongful_declines_max"]),
        ("field_accuracy", ">=", th["field_accuracy_min"]),
        ("retrieval_hit_at_3", ">=", th["retrieval_hit_at_3_min"]),
        ("citation_coverage", ">=", th["citation_coverage_min"]),
        ("consistency", ">=", th["consistency_min"]),
    ]
    for key, op, bound in checks:
        v = results[key]
        ok = v >= bound if op == ">=" else v <= bound
        if not ok:
            fails.append(f"{key}={v} (required {op} {bound})")
    return fails


def to_markdown(r: dict, fails: list[str]) -> str:
    pct = lambda x: f"{x:.1%}"  # noqa: E731
    lines = [
        "# Eval report", "",
        f"**Release gate: {'PASS' if not fails else 'FAIL'}** on {r['n_cases']} golden cases", "",
        "| Metric | Result |", "|---|---|",
        f"| Decision accuracy vs. senior underwriter | {pct(r['decision_accuracy'])} |",
        f"| Unsafe accepts (should have been referred/declined) | {r['unsafe_accepts']} |",
        f"| Wrongful declines | {r['wrongful_declines']} |",
        f"| Over-referrals (safe, costs reviewer time) | {r['over_referrals']} |",
        f"| Risk capture rate (problem accounts not accepted) | {pct(r['risk_capture_rate'])} |",
        f"| Field extraction accuracy | {pct(r['field_accuracy'])} |",
        f"| Loss ratio mean absolute error | {r['loss_ratio_mean_abs_error_pts']} pts |",
        f"| Guideline retrieval hit@1 / hit@3 | {pct(r['retrieval_hit_at_1'])} / {pct(r['retrieval_hit_at_3'])} |",
        f"| Free-text guideline Q&A hit@1 / hit@3 ({r['retrieval_qa']['n']} questions) | "
        f"{pct(r['retrieval_qa']['hit_at_1'])} / {pct(r['retrieval_qa']['hit_at_3'])} |",
        f"| Warnings with a citation | {pct(r['citation_coverage'])} |",
        f"| Consistency (identical memo on re-run) | {pct(r['consistency'])} |",
        f"| Messy-document cases handled safely | {r['messy_cases_handled_safely']} of {r['messy_document_cases']} |",
        f"| Pipeline latency, median / p95 | {r['latency_ms_median']} ms / {r['latency_ms_p95']} ms |",
        "", "## Misses", "",
    ]
    if r["errors"]:
        lines += ["| Case | Truth | Copilot | Kind | Document issues |", "|---|---|---|---|---|"]
        lines += [f"| {e['case_id']} | {e['truth']} | {e['copilot']} | {e['kind']} | {', '.join(e['quirks']) or '-'} |"
                  for e in r["errors"]]
    else:
        lines.append("None.")
    if fails:
        lines += ["", "## Gate failures", ""] + [f"- {f}" for f in fails]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()
    results = run()
    fails = gate(results, json.loads((ROOT / "evals" / "thresholds.json").read_text()))
    (ROOT / "evals" / "results.json").write_text(json.dumps(results, indent=2))
    report = to_markdown(results, fails)
    (ROOT / "evals" / "report.md").write_text(report)
    print(report)
    if args.gate and fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
