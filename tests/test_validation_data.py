"""Tests for the checksum-pinned real-data preparation contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import fetch_validation_data as fetch


ROOT = Path(__file__).resolve().parents[1]


def test_source_manifest_has_unique_pinned_downloads() -> None:
    manifest = json.loads((ROOT / "validation/data_sources.json").read_text())
    sources = manifest["sources"]

    assert manifest["reference_environment"] == {
        "r": "4.6.0",
        "deseq2": "1.52.0",
        "apeglm": "1.34.0",
        "biocparallel": "1.46.0",
        "summarizedexperiment": "1.42.0",
    }
    assert len(sources) == 3
    assert len({source["id"] for source in sources}) == len(sources)
    assert len({source["filename"] for source in sources}) == len(sources)
    for source in sources:
        assert str(source["url"]).startswith("https://")
        assert int(source["size_bytes"]) > 0
        assert len(str(source["sha256"])) == 64
        int(str(source["sha256"]), 16)


def test_download_verification_rejects_size_and_hash_mismatches(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.bin"
    path.write_bytes(b"reproducible")
    source = {
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    fetch.verify(path, source)

    with pytest.raises(RuntimeError, match="expected .* bytes"):
        fetch.verify(path, {**source, "size_bytes": path.stat().st_size + 1})
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        fetch.verify(path, {**source, "sha256": "0" * 64})
    assert fetch.display(path) == str(path)


def test_prepared_manifest_links_sources_and_records_gtex_selection() -> None:
    source_path = ROOT / "validation/data_sources.json"
    prepared = json.loads(
        (ROOT / "validation/prepared_data_manifest.json").read_text()
    )

    assert prepared["source_manifest"] == "validation/data_sources.json"
    assert prepared["source_manifest_sha256"] == hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    assert len(prepared["cases"]) == 6
    gtex = prepared["cases"]["gtex_blood_muscle"]["metadata"]
    assert gtex["sample_seed"] == 1
    assert gtex["samples_per_tissue"] == 150
    assert len(gtex["selected_source_columns"]) == 300
    assert len(gtex["selected_source_samples"]) == 300
