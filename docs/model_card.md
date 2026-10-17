# Model card: large-loss propensity model

**Purpose.** Estimate the probability that a commercial property/GL account has a
large loss in the next policy year. Advisory only: at 25% or higher it refers
an otherwise clean account to a senior underwriter. It never declines.

**Type.** Logistic regression on standardized features (scikit-learn). Chosen
over gradient boosting for transparency: each prediction decomposes into
per-feature contributions shown in the memo.

**Training data.** 6,000 synthetic policies from `data/synth/generate.py`
(4,500 train / 1,500 holdout, stratified). Large-loss base rate 12.8%. The
generating process includes an interaction (old building without sprinklers)
that the linear model does not capture, so the model is deliberately imperfect.

**Holdout performance.** AUC 0.80, Brier score 0.091.

**Features and standardized coefficients** (positive raises risk):

| Feature | Coefficient |
|---|---|
| Prior claim count (5 yrs) | +0.63 |
| Building age | +0.59 |
| High-hazard class | +0.49 |
| Medium-hazard class | +0.26 |
| Fire protection class | +0.24 |
| Property values (log TIV) | +0.11 |
| Revenue size (log) | +0.08 |
| Sprinkler protection | −0.41 |
| Years in business | −0.45 |

**Limitations.** Trained on synthetic data, so the numbers show the method, not
real-world risk. In production, retrain on the carrier's own policy and claims
history, check calibration by segment, and monitor drift quarterly.

**Excluded inputs.** No protected characteristics and no ZIP-level geography.
See `docs/governance.md`.
