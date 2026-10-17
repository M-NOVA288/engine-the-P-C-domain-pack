"""Risk scoring agent: an interpretable large-loss propensity model.

A logistic regression is used on purpose. Every prediction decomposes into
per-feature contributions an underwriter (and a regulator) can read, which
matters more in insurance than squeezing out a little extra AUC.
"""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np

from copilot.engine.schema import CaseFile, Finding

from .config import AS_OF_YEAR, hazard_of

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "risk_model.joblib"

FEATURES = [
    "hazard_medium", "hazard_high", "log_tiv", "protection_class", "sprinklered",
    "building_age", "prior_claims_5y", "years_in_business", "log_revenue",
]
LABELS = {
    "hazard_medium": "Medium-hazard class", "hazard_high": "High-hazard class",
    "log_tiv": "Property values (TIV)", "protection_class": "Fire protection class",
    "sprinklered": "Sprinkler protection", "building_age": "Building age",
    "prior_claims_5y": "Prior claim count", "years_in_business": "Years in business",
    "log_revenue": "Revenue size",
}


def build_features(rec: dict) -> list[float] | None:
    """rec keys: naics, tiv, protection_class, all_sprinklered, oldest_year_built,
    claim_count_5y, years_in_business, annual_revenue."""
    needed = ["tiv", "protection_class", "all_sprinklered", "oldest_year_built",
              "claim_count_5y", "years_in_business", "annual_revenue"]
    if any(rec.get(k) is None for k in needed):
        return None
    hz = hazard_of(rec.get("naics"))
    return [
        1.0 if hz == "medium" else 0.0,
        1.0 if hz == "high" else 0.0,
        math.log10(max(rec["tiv"], 1)),
        float(rec["protection_class"]),
        1.0 if rec["all_sprinklered"] else 0.0,
        float(AS_OF_YEAR - rec["oldest_year_built"]),
        float(min(rec["claim_count_5y"], 10)),
        float(min(rec["years_in_business"], 25)),
        math.log10(max(rec["annual_revenue"], 1)),
    ]


_model = None


def _load():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Risk model not trained. Run `python -m models.train` first.")
        _model = joblib.load(MODEL_PATH)
    return _model


def explain(x: list[float]) -> list[dict]:
    pipe = _load()
    scaler, lr = pipe.named_steps["scale"], pipe.named_steps["clf"]
    z = (np.array(x) - scaler.mean_) / scaler.scale_
    contrib = z * lr.coef_[0]
    order = np.argsort(-np.abs(contrib))
    return [{"feature": LABELS[FEATURES[i]], "contribution": round(float(contrib[i]), 3)} for i in order[:4]]


def score(case: CaseFile) -> CaseFile:
    rec = {k: case.value(k) for k in ["naics", "tiv", "protection_class", "all_sprinklered",
                                      "oldest_year_built", "years_in_business", "annual_revenue"]}
    rec["claim_count_5y"] = case.metrics.get("claim_count_5y")
    x = build_features(rec)
    if x is None:
        case.findings.append(Finding(step="scoring", severity="warn",
                                     message="Insufficient data to compute the modeled risk score."))
        return case
    p = float(_load().predict_proba([x])[0, 1])
    case.risk_score = round(p, 4)
    case.risk_drivers = explain(x)
    drivers = ", ".join(f"{d['feature']} ({'+' if d['contribution'] > 0 else ''}{d['contribution']})"
                        for d in case.risk_drivers[:3])
    case.findings.append(Finding(step="scoring", severity="info",
                                 message=f"Modeled large-loss probability {p:.0%}. Top drivers: {drivers}."))
    return case
