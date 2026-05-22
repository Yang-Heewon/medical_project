"""Paired statistical tests."""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_ttest(a, b) -> dict:
    """같은 inference sample에 대한 paired t-test."""
    res = stats.ttest_rel(np.asarray(a), np.asarray(b), nan_policy="omit")
    return {"test": "paired_ttest", "statistic": float(res.statistic), "pvalue": float(res.pvalue)}


def wilcoxon_test(a, b) -> dict:
    """비정규/소표본용 Wilcoxon signed-rank test."""
    res = stats.wilcoxon(np.asarray(a), np.asarray(b))
    return {"test": "wilcoxon", "statistic": float(res.statistic), "pvalue": float(res.pvalue)}
