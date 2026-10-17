"""Run the copilot on one submission folder and print the memo.

    python -m copilot.cli data/submissions/CASE-0001
    python -m copilot.cli data/submissions/CASE-0001 --approve accept --reviewer "J. Smith"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from copilot.engine import Decision, Orchestrator
from copilot.packs import get_pack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--pack", default="pnc_underwriting")
    ap.add_argument("--approve", choices=[d.value for d in Decision])
    ap.add_argument("--reviewer", default="demo-underwriter")
    ap.add_argument("--json", action="store_true", help="print the full case file as JSON")
    args = ap.parse_args()

    docs = {f.name: f.read_text() for f in sorted(args.folder.iterdir()) if f.suffix == ".txt"}
    orch = Orchestrator(get_pack(args.pack))
    case = orch.run(args.folder.name, docs)
    if args.approve:
        orch.approve(case, args.reviewer, Decision(args.approve))

    out = Path("out")
    out.mkdir(exist_ok=True)
    (out / f"{case.case_id}_memo.md").write_text(case.memo_markdown)
    if args.json:
        print(json.dumps(case.model_dump(mode="json"), indent=2, default=str))
    else:
        print(case.memo_markdown)
        print(f"\n[timings ms] {case.timings_ms}")


if __name__ == "__main__":
    main()
