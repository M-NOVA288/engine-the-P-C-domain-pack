"""Intake agent: turn messy submission documents into typed, cited fields.

Local mode parses the synthetic text documents with tolerant regexes. In Azure
mode, swap `read_documents` for Azure AI Document Intelligence (prebuilt layout
or a custom model trained on ACORD 125/140) and keep the same field contract.
"""
from __future__ import annotations

import re

from copilot.engine.schema import CaseFile, Citation, ExtractedField

from .config import AS_OF_YEAR

MONEY_RE = re.compile(r"\$?\s*([\d,]*\.?\d+)\s*(million|mm|m|k|thousand)?\b", re.I)


def parse_money(text: str) -> float | None:
    m = MONEY_RE.search(text)
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    if unit in ("m", "mm", "million"):
        num *= 1_000_000
    elif unit in ("k", "thousand"):
        num *= 1_000
    return num


def _find_line(doc: str, patterns: list[str]) -> tuple[int, str, str] | None:
    """Return (line_no, label, value_text) for the first label that matches."""
    for i, line in enumerate(doc.splitlines(), start=1):
        for pat in patterns:
            m = re.match(rf"^\s*{pat}\s*[:\-]\s*(.+)$", line, re.I)
            if m:
                return i, pat, m.group(1).strip()
    return None


def _field(name, value, conf, source, line_no, snippet) -> ExtractedField:
    return ExtractedField(
        name=name, value=value, confidence=conf,
        citation=Citation(source=source, locator=f"line {line_no}", snippet=snippet[:120]),
    )


def extract_application(case: CaseFile) -> None:
    doc = case.documents.get("application.txt", "")
    src = "application.txt"

    hit = _find_line(doc, [r"Applicant Name", r"Named Insured", r"Insured"])
    if hit:
        case.fields["applicant_name"] = _field("applicant_name", hit[2], 0.98, src, hit[0], hit[2])

    hit = _find_line(doc, [r"Mailing State", r"State of Domicile", r"State"])
    if hit:
        state = re.sub(r"[^A-Za-z]", "", hit[2])[:2].upper()
        case.fields["state"] = _field("state", state, 0.97, src, hit[0], hit[2])

    hit = _find_line(doc, [r"NAICS Code", r"NAICS", r"Primary NAICS"])
    if hit:
        m = re.search(r"\b(\d{6})\b", hit[2])
        if m:
            case.fields["naics"] = _field("naics", m.group(1), 0.99, src, hit[0], hit[2])
        else:
            # Likely OCR damage (l for 1, O for 0). Repair it, but at low confidence so
            # the case is referred for a human to confirm instead of silently trusted.
            repaired = hit[2].translate(str.maketrans({"l": "1", "I": "1", "O": "0", "o": "0"}))
            m = re.search(r"\b(\d{6})\b", repaired)
            if m:
                case.fields["naics"] = _field("naics", m.group(1), 0.6, src, hit[0], hit[2])

    hit = _find_line(doc, [r"Years in Business", r"Yrs in Business"])
    if hit and re.search(r"\d+", hit[2]):
        yib = int(re.search(r"\d+", hit[2]).group())
        case.fields["years_in_business"] = _field("years_in_business", yib, 0.97, src, hit[0], hit[2])
    else:
        hit = _find_line(doc, [r"Year Established", r"Date Business Started", r"Established"])
        if hit and re.search(r"(19|20)\d{2}", hit[2]):
            year = int(re.search(r"(19|20)\d{2}", hit[2]).group())
            # Derived value, so slightly lower confidence than a direct statement.
            case.fields["years_in_business"] = _field(
                "years_in_business", AS_OF_YEAR - year, 0.9, src, hit[0], hit[2])

    hit = _find_line(doc, [r"Annual Revenue", r"Revenues?", r"Gross Sales", r"Annual Sales"])
    if hit:
        val = parse_money(hit[2])
        if val:
            case.fields["annual_revenue"] = _field("annual_revenue", val, 0.95, src, hit[0], hit[2])
    else:
        # Fall back to free-text remarks. Useful to show the underwriter, but low
        # confidence because the figure is unverified ("roughly", "last year").
        for i, line in enumerate(doc.splitlines(), start=1):
            m = re.search(r"(sales|revenue)s?\b[^$\d]{0,30}(\$?\s*[\d,.]+\s*(?:million|m|k)?)", line, re.I)
            if line.lower().startswith("remarks") and m:
                val = parse_money(m.group(2))
                if val:
                    case.fields["annual_revenue"] = _field("annual_revenue", val, 0.6, src, i, line.strip())
                break

    hit = _find_line(doc, [r"Employees", r"Number of Employees", r"FTEs?"])
    if hit and re.search(r"\d+", hit[2]):
        case.fields["employees"] = _field(
            "employees", int(re.search(r"\d+", hit[2]).group()), 0.95, src, hit[0], hit[2])


def _yes(text: str) -> bool:
    t = text.strip().lower()
    return t.startswith("y") or t.startswith("100") or t == "full"


def extract_sov(case: CaseFile) -> None:
    doc = case.documents.get("sov.txt", "")
    if not doc:
        return
    lines = doc.splitlines()
    header_idx = next((i for i, l in enumerate(lines) if "|" in l and "building" in l.lower()), None)
    if header_idx is None:
        return
    headers = [h.strip().lower() for h in lines[header_idx].split("|")]

    def col(*names):
        for n in names:
            for i, h in enumerate(headers):
                if n in h:
                    return i
        return None

    c_bldg, c_cont, c_bi = col("building"), col("contents", "bpp"), col("bi", "business income")
    c_year, c_spr, c_pc = col("year built", "yr built"), col("sprink"), col("protection", "pc")

    tiv, years, pcs, sprinklered = 0.0, [], [], []
    rows = 0
    for line in lines[header_idx + 1:]:
        if "|" not in line or set(line.strip()) <= {"-", "|", " "}:
            continue
        cells = [c.strip() for c in line.split("|")]
        # Skip summary rows; summing them again doubles TIV (a bug the evals caught).
        if re.match(r"^(total|totals|grand total|subtotal)\b", cells[0], re.I):
            continue
        rows += 1
        for c in (c_bldg, c_cont, c_bi):
            if c is not None and c < len(cells):
                tiv += parse_money(cells[c]) or 0
        if c_year is not None and re.search(r"(18|19|20)\d{2}", cells[c_year]):
            years.append(int(re.search(r"(18|19|20)\d{2}", cells[c_year]).group()))
        if c_pc is not None and re.search(r"\d+", cells[c_pc]):
            pcs.append(max(int(x) for x in re.findall(r"\d+", cells[c_pc])))
        if c_spr is not None:
            sprinklered.append(_yes(cells[c_spr]))

    if rows == 0:
        return
    src = "sov.txt"
    snip = f"{rows} location(s)"
    case.fields["tiv"] = _field("tiv", tiv, 0.96, src, header_idx + 1, snip)
    case.fields["locations"] = _field("locations", rows, 0.99, src, header_idx + 1, snip)
    if years:
        case.fields["oldest_year_built"] = _field("oldest_year_built", min(years), 0.95, src, header_idx + 1, snip)
    if pcs:
        case.fields["protection_class"] = _field("protection_class", max(pcs), 0.95, src, header_idx + 1, snip)
    if sprinklered:
        case.fields["all_sprinklered"] = _field("all_sprinklered", all(sprinklered), 0.95, src, header_idx + 1, snip)


def extract_loss_run(case: CaseFile) -> None:
    doc = case.documents.get("loss_run.txt", "")
    if not doc:
        return
    src = "loss_run.txt"
    years, claims = [], []
    section = None
    for i, line in enumerate(doc.splitlines(), start=1):
        low = line.lower()
        if "policy year" in low and "|" in line:
            section = "years"
            continue
        if "date of loss" in low and "|" in line:
            section = "claims"
            continue
        if "|" not in line or set(line.strip()) <= {"-", "|", " "}:
            continue
        cells = [c.strip() for c in line.split("|")]
        if section == "years" and len(cells) >= 4:
            years.append({"year": cells[0], "earned_premium": parse_money(cells[1]) or 0.0,
                          "claims": int(re.sub(r"\D", "", cells[2]) or 0),
                          "incurred": parse_money(cells[-1]) or 0.0, "line": i})
        elif section == "claims" and len(cells) >= 3:
            claims.append({"date": cells[0], "cause": cells[1],
                           "incurred": parse_money(cells[2]) or 0.0, "line": i})

    case.fields["loss_years"] = _field("loss_years", len(years), 0.98, src, 1, f"{len(years)} policy years")
    case.fields["loss_run_years_detail"] = ExtractedField(name="loss_run_years_detail", value=years)
    case.fields["loss_run_claims"] = ExtractedField(name="loss_run_claims", value=claims)


def extract(case: CaseFile) -> CaseFile:
    extract_application(case)
    extract_sov(case)
    extract_loss_run(case)
    return case
