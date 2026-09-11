from __future__ import annotations

import numpy as np

from bench.score_gtex_scaling import (
    jaccard_005,
    max_relative,
    p95_absolute,
    p95_relative,
    pearson,
    spearman,
)


def test_scaling_metrics_ignore_paired_nonfinite_values() -> None:
    observed = np.array([1.0, 2.0, np.nan, 4.0])
    reference = np.array([1.0, 2.5, 3.0, np.nan])
    assert np.isclose(max_relative(observed, reference), 0.2)
    assert np.isclose(p95_relative(observed, reference), 0.19)
    assert np.isclose(p95_absolute(observed, reference), 0.475)


def test_jaccard_uses_finite_adjusted_pvalues_below_threshold() -> None:
    observed = np.array([0.01, 0.2, np.nan, 0.04])
    reference = np.array([0.01, 0.03, 0.02, np.nan])
    assert jaccard_005(observed, reference) == 1 / 4


def test_correlations_are_one_for_affine_monotone_values() -> None:
    observed = np.array([1.0, 2.0, 4.0, np.nan])
    reference = np.array([3.0, 5.0, 9.0, 11.0])
    assert np.isclose(pearson(observed, reference), 1.0)
    assert np.isclose(spearman(observed, reference), 1.0)
