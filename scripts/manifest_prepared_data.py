"""Write a content manifest for deterministic validation inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation/data"
SOURCE_MANIFEST = ROOT / "validation/data_sources.json"
CASES = (
    "airway",
    "airway_cell",
    "airway_dex",
    "gtex_blood_muscle",
    "pasilla",
    "pasilla_2fac",
)
FILES = ("counts.csv", "coldata.csv", "meta.json")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def build_manifest(data: Path = DATA) -> dict[str, object]:
    records: dict[str, dict[str, object]] = {}
    for case in CASES:
        records[case] = {"files": {}}
        for filename in FILES:
            path = data / case / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"missing prepared input {path}; run validation/prepare_inputs.R"
                )
            records[case]["files"][filename] = {
                "size_bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        records[case]["metadata"] = json.loads(
            (data / case / "meta.json").read_text()
        )
    return {
        "schema_version": 1,
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": digest(SOURCE_MANIFEST),
        "cases": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation/prepared_data_manifest.json",
    )
    args = parser.parse_args()

    output = build_manifest(args.data.resolve())
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
