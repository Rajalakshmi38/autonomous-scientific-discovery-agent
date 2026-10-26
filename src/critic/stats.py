"""Statistical helpers for the critic: plain functions on lists of numbers.

No language model is involved anywhere in this file. Every number comes from SciPy/NumPy
calculations that the tests check against known textbook values.

What is computed for a comparison of two groups ``a`` (baseline) and ``b`` (treatment):
  * primary test: Welch's t-test (unpaired) or paired t-test (paired=True)
  * robustness check: Mann-Whitney U (unpaired) or Wilcoxon signed-rank (paired)
  * resampled scores (repeated random train/test splits of one dataset) are NOT independent, so for them the
    corrected resampled t-test of Nadeau and Bengio (2003) replaces the plain tests
  * a confidence interval for the mean difference (b minus a)
  * an effect size: Hedges' g (unpaired) or dz (paired)
A p-value says how surprising the data would be if there were no difference. It is NOT the
probability that the hypothesis is true.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy import stats


class StatsInputError(ValueError):
    """The data cannot be analysed (too few values, no variation, non-finite numbers, ...)."""


@dataclass
class ComparisonResult:
    paired: bool
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    mean_diff: float                 # mean(b) - mean(a)
    ci_low: float
    ci_high: float
    ci_level: float
    primary_test: str
    primary_statistic: float
    primary_p: float
    robust_test: str
    robust_p: Optional[float]        # None when no robustness test applies
    effect_size: float
    effect_size_name: str
    notes: List[str] = field(default_factory=list)


def _clean(values: Sequence[float], label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise StatsInputError(f"group {label} must be a flat list of numbers")
    if arr.size < 2:
        raise StatsInputError(f"group {label} needs at least 2 values (got {arr.size})")
    if not np.all(np.isfinite(arr)):
        raise StatsInputError(f"group {label} contains NaN or infinite values")
    return arr


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (controls the chance of any false positive across k tests)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted


def compare_groups(
    a: Sequence[float], b: Sequence[float], paired: bool = False, alpha: float = 0.05
) -> ComparisonResult:
    """Compare baseline ``a`` with treatment ``b``. Raises StatsInputError if the data cannot be analysed."""
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5")
    x, y = _clean(a, "a"), _clean(b, "b")
    level = 1 - alpha
    notes: List[str] = []

    if paired:
        if x.size != y.size:
            raise StatsInputError(f"paired data needs equal group sizes (got {x.size} and {y.size})")
        diff = y - x
        sd = float(np.std(diff, ddof=1))
        if sd == 0:
            raise StatsInputError("all paired differences are identical, so the test is undefined")
        n = diff.size
        res = stats.ttest_rel(y, x)
        se = sd / math.sqrt(n)
        half = float(stats.t.ppf(1 - alpha / 2, n - 1)) * se
        mean_diff = float(np.mean(diff))
        robust = stats.wilcoxon(y, x)
        return ComparisonResult(
            True, n, n, float(np.mean(x)), float(np.mean(y)), mean_diff, mean_diff - half, mean_diff + half, level,
            "paired t-test", float(res.statistic), float(res.pvalue),
            "Wilcoxon signed-rank", float(robust.pvalue),
            mean_diff / sd, "dz (mean difference / sd of differences)", notes,
        )

    va, vb = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    if va == 0 and vb == 0:
        raise StatsInputError("both groups have zero variance, so the test is undefined")
    na, nb = x.size, y.size
    res = stats.ttest_ind(y, x, equal_var=False)
    se = math.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    half = float(stats.t.ppf(1 - alpha / 2, df)) * se
    mean_diff = float(np.mean(y) - np.mean(x))
    robust = stats.mannwhitneyu(y, x, alternative="two-sided")
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    correction = 1 - 3 / (4 * (na + nb - 2) - 1)
    return ComparisonResult(
        False, na, nb, float(np.mean(x)), float(np.mean(y)), mean_diff, mean_diff - half, mean_diff + half, level,
        "Welch t-test", float(res.statistic), float(res.pvalue),
        "Mann-Whitney U", float(robust.pvalue),
        (mean_diff / pooled) * correction, "Hedges' g", notes,
    )


def compare_resampled(
    a: Sequence[float], b: Sequence[float], n_train: int, n_test: int, alpha: float = 0.05
) -> ComparisonResult:
    """Paired scores from k repeated random train/test splits of one dataset.

    Scores from overlapping splits are correlated, so the plain paired t-test is far too optimistic.
    Nadeau and Bengio (2003) correct the variance: var_corrected = (1/k + n_test/n_train) * var(differences).
    """
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5")
    x, y = _clean(a, "a"), _clean(b, "b")
    if x.size != y.size:
        raise StatsInputError(f"paired scores need equal group sizes (got {x.size} and {y.size})")
    if n_train < 1 or n_test < 1:
        raise StatsInputError("n_train and n_test must be positive")
    diff = y - x
    k = diff.size
    mean_diff = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    name = "corrected resampled t-test (Nadeau-Bengio)"
    effect_name = "dz (mean difference / sd of differences)"
    if sd == 0:
        if mean_diff != 0:
            raise StatsInputError("every resample gave exactly the same non-zero difference, so the variance is zero")
        return ComparisonResult(True, k, k, float(np.mean(x)), float(np.mean(y)), 0.0, 0.0, 0.0, 1 - alpha,
                                name, 0.0, 1.0, "not used (resampled scores are dependent)", None, 0.0, effect_name,
                                ["Both methods scored identically in every resample."])
    se = math.sqrt((1.0 / k + n_test / n_train) * sd ** 2)
    t_value = mean_diff / se
    p_value = float(2 * stats.t.sf(abs(t_value), k - 1))
    half = float(stats.t.ppf(1 - alpha / 2, k - 1)) * se
    return ComparisonResult(True, k, k, float(np.mean(x)), float(np.mean(y)), mean_diff, mean_diff - half,
                            mean_diff + half, 1 - alpha, name, float(t_value), p_value,
                            "not used (resampled scores are dependent)", None, mean_diff / sd, effect_name)
