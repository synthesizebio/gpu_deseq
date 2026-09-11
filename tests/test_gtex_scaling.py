from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bench.gtex_scaling import CASE_NAMES, case_spec, stable_order


def _samples() -> pd.DataFrame:
    paper = json.loads(
        (Path(__file__).parents[1] / "validation/data/gtex_blood_muscle/meta.json").read_text()
    )
    columns = paper["selected_source_columns"]
    rows = []
    for index, column in enumerate(columns):
        rows.append(
            {
                "source_column": column,
                "sample_id": f"paper-{column}",
                "tissue": "Whole Blood" if index < 150 else "Muscle - Skeletal",
            }
        )
    next_column = max(columns) + 1
    for tissue, count in (
        ("Whole Blood", 306),
        ("Muscle - Skeletal", 325),
        ("Skin", 397),
        ("Adipose", 386),
        ("Lung", 374),
        ("Artery", 363),
        ("Thyroid", 361),
        ("Nerve", 335),
        ("Esophagus", 331),
        ("Cells", 306),
        ("Heart", 283),
        ("Pancreas", 271),
    ):
        for offset in range(count):
            rows.append(
                {
                    "source_column": next_column,
                    "sample_id": f"extra-{next_column}-{offset}",
                    "tissue": tissue,
                }
            )
            next_column += 1
    return pd.DataFrame(rows)


def test_stable_order_is_input_order_independent() -> None:
    rows = _samples().iloc[:40]
    forward = stable_order(rows)
    reverse = stable_order(rows.iloc[::-1])
    assert forward == reverse


@pytest.mark.parametrize("name", CASE_NAMES)
def test_case_specs_are_deterministic_and_well_formed(name: str) -> None:
    samples = _samples()
    first = case_spec(name, samples)
    second = case_spec(name, samples.sample(frac=1, random_state=17))
    assert first == second
    assert first["P"] == len(first["tissues"])
    assert first["n_samples"] == len(first["labels"])
    assert first["n_samples"] == len(first["source_columns_zero_based"])
    assert first["n_samples"] == sum(first["group_counts"].values())
    assert len(set(first["source_columns_zero_based"])) == first["n_samples"]


def test_original_paper_cohort_is_exact_prefix() -> None:
    paper = json.loads(
        (Path(__file__).parents[1] / "validation/data/gtex_blood_muscle/meta.json").read_text()
    )
    expected = [int(value) - 1 for value in paper["selected_source_columns"]]
    spec = case_spec("p2_300", _samples())
    assert spec["source_columns_zero_based"] == expected


def test_balanced_two_tissue_maximum_is_912() -> None:
    spec = case_spec("p2_912", _samples())
    assert spec["P"] == 2
    assert spec["n_samples"] == 912
    assert set(spec["group_counts"].values()) == {456}
