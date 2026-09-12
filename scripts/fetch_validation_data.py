"""Download and verify the immutable source files used by paper benchmarks.

This script uses only the Python standard library so it can run before the
project or R environment is installed. Prepared CSV inputs are deliberately
not stored in Git; ``validation/prepare_inputs.R`` deterministically derives
them from these checksum-pinned sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "validation/data_sources.json"
DEFAULT_DESTINATION = ROOT / "validation/sources"
BUFFER_SIZE = 8 * 1024 * 1024


def display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, source: dict[str, object]) -> None:
    expected_size = int(source["size_bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"{path.name}: expected {expected_size} bytes, found {actual_size}"
        )
    expected_hash = str(source["sha256"])
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{path.name}: SHA-256 mismatch\n"
            f"  expected {expected_hash}\n"
            f"  found    {actual_hash}"
        )


def download(source: dict[str, object], destination: Path, force: bool) -> None:
    target = destination / str(source["filename"])
    if target.exists() and not force:
        verify(target, source)
        print(f"verified {display(target)}")
        return

    partial = target.with_suffix(target.suffix + ".part")
    partial.unlink(missing_ok=True)
    print(f"downloading {source['url']} -> {display(target)}", flush=True)
    request = urllib.request.Request(
        str(source["url"]), headers={"User-Agent": "gpu-deseq-data-fetch/1"}
    )
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=BUFFER_SIZE)
        verify(partial, source)
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print(f"verified {display(target)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify existing files without downloading missing sources",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for source in manifest["sources"]:
        target = destination / source["filename"]
        if args.verify_only:
            if not target.exists():
                raise FileNotFoundError(f"missing source: {target}")
            verify(target, source)
            print(f"verified {target}")
        else:
            download(source, destination, args.force)

    print(f"all {len(manifest['sources'])} source files match {MANIFEST}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as error:
        print(f"data fetch failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
