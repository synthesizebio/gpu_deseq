"""Copy versioned ``run_r.R`` timing records into the benchmark artifact."""
from __future__ import annotations

import json
from pathlib import Path


CACHE = Path("bench/cache")
DESTINATION = Path("bench/results/timings.json")


def main() -> None:
    timings = json.loads(DESTINATION.read_text())
    current = {}
    for record in sorted(CACHE.glob("*/r_timings.json")):
        value = json.loads(record.read_text())
        current[value["case"]] = value
    timings["r"] = current
    DESTINATION.write_text(json.dumps(timings, indent=2) + "\n")
    print(f"updated {len(current)} R timing records in {DESTINATION}")


if __name__ == "__main__":
    main()
