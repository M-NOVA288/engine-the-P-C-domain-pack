"""Append-only audit trail: one JSON line per pipeline step and per human action.

Regulators and model-risk teams ask "why did the system say that, and who
approved it?" This log answers both without re-running anything.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class AuditLog:
    def __init__(self, path: Path | str = "audit/audit.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, case_id: str, step: str, payload: dict[str, Any]) -> None:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "case_id": case_id,
            "step": step,
            "payload": payload,
        }
        entry["hash"] = hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode()).hexdigest()[:16]
        with self.path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
