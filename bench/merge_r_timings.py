"""Merge completed staged R timing records into the retained timing bundle.

This intentionally replaces only ``timings.json["r"]``. GPU timings and all
other retained benchmark content remain untouched.
"""
from __future__ import annotations

import json
from pathlib import Path


RESULTS = Path("bench/results/timings.json")
CACHE = Path("bench/cache")


def main() -> None:
    timings = json.loads(RESULTS.read_text())
    records = {}
    for path in sorted(CACHE.glob("*/r_timings.json")):
        record = json.loads(path.read_text())
        if record.get("reps") != 5:
            raise RuntimeError(f"{path} has reps={record.get('reps')}, expected 5")
        if not record.get("output_equivalent_to_DESeq"):
            raise RuntimeError(f"{path} did not pass the DESeq2 equivalence gate")
        records[record["case"]] = record
    if set(records) != set(timings["r"]):
        raise RuntimeError(
            f"case mismatch: cache={sorted(records)}, retained={sorted(timings['r'])}"
        )
    timings["r"] = records
    RESULTS.write_text(json.dumps(timings, indent=2) + "\n")
    print(f"updated only the R timing section in {RESULTS}")


if __name__ == "__main__":
    main()
