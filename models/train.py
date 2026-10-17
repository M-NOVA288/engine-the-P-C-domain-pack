"""Train the large-loss propensity model and write a metrics card.

    python -m models.train
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from copilot.packs.pnc.scoring import FEATURES, MODEL_PATH, build_features

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "portfolio.csv", dtype={"naics": str})
    X = [build_features(r) for r in df.to_dict("records")]
    y = df["large_loss_next_year"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]

    metrics = {
        "n_train": len(X_tr), "n_test": len(X_te), "base_rate": round(float(y.mean()), 4),
        "auc": round(float(roc_auc_score(y_te, p)), 4),
        "brier": round(float(brier_score_loss(y_te, p)), 4),
        "coefficients": {f: round(float(c), 3) for f, c in zip(FEATURES, pipe.named_steps["clf"].coef_[0])},
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    (ROOT / "models" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model. Holdout AUC {metrics['auc']}, Brier {metrics['brier']}, base rate {metrics['base_rate']:.1%}")


if __name__ == "__main__":
    main()
