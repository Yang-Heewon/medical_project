"""TextGrad prompt optimization gate utilities.

핵심 원칙:
- MedGemma weight는 절대 수정하지 않는다.
- STYLE_PROFILE만 TextGrad variable로 둔다.
- Qwen critic은 candidate prompt를 제안한다.
- candidate prompt가 impression metric을 올려도 lesion preservation statistical gate를 통과하지 못하면 reject한다.

통계 해석:
- 단순 paired t-test에서 p > 0.05라고 해서 "동등하다"가 증명되는 것은 아니다.
- 병변 정확도를 baseline과 유사하게 보존하려면 non-inferiority 또는 equivalence margin을 먼저 정해야 한다.
- 기본 권장값은 non-inferiority다. 즉, optimized prompt가 baseline보다 margin 이상 나빠지지 않으면 통과한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _paired_arrays(baseline_scores: Iterable[float], candidate_scores: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """paired score vector를 finite float array로 정리한다."""
    baseline = np.asarray(list(baseline_scores), dtype=float)
    candidate = np.asarray(list(candidate_scores), dtype=float)
    if baseline.shape != candidate.shape:
        raise ValueError(f"paired score length mismatch: baseline={len(baseline)}, candidate={len(candidate)}")
    mask = np.isfinite(baseline) & np.isfinite(candidate)
    return baseline[mask], candidate[mask]


def _ttest_rel(candidate: np.ndarray, baseline: np.ndarray) -> tuple[float | None, float | None]:
    """scipy paired t-test wrapper. scipy가 없거나 n이 작으면 None을 반환한다."""
    if len(candidate) < 2:
        return None, None
    try:
        from scipy import stats

        res = stats.ttest_rel(candidate, baseline, nan_policy="omit")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return None, None


def _t_cdf(value: float, df: int) -> float:
    from scipy import stats

    return float(stats.t.cdf(value, df))


def _t_ppf(prob: float, df: int) -> float:
    from scipy import stats

    return float(stats.t.ppf(prob, df))


def evaluate_lesion_preservation_ttest(
    baseline_scores: Iterable[float],
    candidate_scores: Iterable[float],
    *,
    margin: float = 0.03,
    alpha: float = 0.05,
    mode: str = "noninferiority",
    min_samples: int = 30,
) -> dict:
    """baseline vs candidate 병변 점수를 paired t-test 계열 gate로 평가한다.

    입력 score는 같은 uid 순서의 per-sample 병변 정확도여야 한다.
    예: human audit correctness, pseudo bbox correctness, lesion label F1, localization score 등.

    mode:
    - noninferiority: candidate가 baseline보다 ``margin`` 이상 나쁘지 않으면 통과.
      H0: mean(candidate - baseline) <= -margin
      H1: mean(candidate - baseline) > -margin
    - equivalence: candidate가 baseline에서 ±margin 안에 있으면 통과.
      TOST: -margin < mean_delta < +margin
    """
    baseline, candidate = _paired_arrays(baseline_scores, candidate_scores)
    mode = str(mode).lower()
    if mode not in {"noninferiority", "equivalence"}:
        raise ValueError(f"Unsupported lesion preservation mode: {mode}")

    diff = candidate - baseline
    n = int(len(diff))
    out = {
        "mode": mode,
        "alpha": float(alpha),
        "margin": float(margin),
        "min_samples": int(min_samples),
        "n": n,
        "baseline_mean": float(np.mean(baseline)) if n else None,
        "candidate_mean": float(np.mean(candidate)) if n else None,
        "mean_delta_candidate_minus_baseline": float(np.mean(diff)) if n else None,
        "paired_ttest_statistic": None,
        "paired_ttest_pvalue_two_sided": None,
        "mean_delta_ci_low_two_sided": None,
        "mean_delta_ci_high_two_sided": None,
        "noninferiority_pvalue": None,
        "equivalence_pvalue_lower": None,
        "equivalence_pvalue_upper": None,
        "lesion_preservation_pass": False,
        "reason": "not_evaluated",
    }

    if n < min_samples:
        out["reason"] = f"too_few_samples: n={n} < min_samples={min_samples}"
        return out
    if n < 2:
        out["reason"] = "too_few_paired_samples"
        return out

    mean_delta = float(np.mean(diff))
    std_delta = float(np.std(diff, ddof=1))
    sem = std_delta / np.sqrt(n)
    df = n - 1

    if sem == 0.0:
        ci_low = ci_high = mean_delta
        if mode == "noninferiority":
            pass_gate = mean_delta > -margin
            out["noninferiority_pvalue"] = 0.0 if pass_gate else 1.0
        else:
            pass_gate = -margin < mean_delta < margin
            out["equivalence_pvalue_lower"] = 0.0 if mean_delta > -margin else 1.0
            out["equivalence_pvalue_upper"] = 0.0 if mean_delta < margin else 1.0
        out["mean_delta_ci_low_two_sided"] = float(ci_low)
        out["mean_delta_ci_high_two_sided"] = float(ci_high)
        out["lesion_preservation_pass"] = bool(pass_gate)
        out["reason"] = "pass" if pass_gate else "failed_margin_with_zero_variance"
        return out

    t_stat, p_two = _ttest_rel(candidate, baseline)
    out["paired_ttest_statistic"] = t_stat
    out["paired_ttest_pvalue_two_sided"] = p_two

    # two-sided CI는 사람이 읽기 좋은 요약이다. 실제 non-inferiority 판정은 one-sided test를 쓴다.
    tcrit_two = _t_ppf(1.0 - alpha / 2.0, df)
    out["mean_delta_ci_low_two_sided"] = float(mean_delta - tcrit_two * sem)
    out["mean_delta_ci_high_two_sided"] = float(mean_delta + tcrit_two * sem)

    # Non-inferiority: mean_delta가 -margin보다 충분히 큰지 검정한다.
    t_noninferiority = (mean_delta + margin) / sem
    p_noninferiority = 1.0 - _t_cdf(t_noninferiority, df)
    out["noninferiority_pvalue"] = float(p_noninferiority)

    if mode == "noninferiority":
        pass_gate = p_noninferiority < alpha
        out["lesion_preservation_pass"] = bool(pass_gate)
        out["reason"] = "pass" if pass_gate else "failed_noninferiority_ttest"
        return out

    # Equivalence / TOST:
    # lower test: mean_delta > -margin, upper test: mean_delta < +margin
    t_lower = (mean_delta + margin) / sem
    p_lower = 1.0 - _t_cdf(t_lower, df)
    t_upper = (mean_delta - margin) / sem
    p_upper = _t_cdf(t_upper, df)
    out["equivalence_pvalue_lower"] = float(p_lower)
    out["equivalence_pvalue_upper"] = float(p_upper)
    pass_gate = p_lower < alpha and p_upper < alpha
    out["lesion_preservation_pass"] = bool(pass_gate)
    out["reason"] = "pass" if pass_gate else "failed_equivalence_tost"
    return out


def accept_optimized_prompt(baseline_metrics: dict, optimized_metrics: dict, rule: dict) -> bool:
    """최적화된 prompt를 채택할지 결정하는 gate."""
    lesion_gate = optimized_metrics.get("lesion_preservation_gate")
    if rule.get("require_lesion_preservation_gate", False):
        if not lesion_gate or not lesion_gate.get("lesion_preservation_pass", False):
            return False

    if optimized_metrics.get("lesion_agreement_rate_vs_baseline", 1.0) < rule.get("min_lesion_agreement_vs_baseline", 0.95):
        return False
    if optimized_metrics.get("clinical_f1", 0.0) < baseline_metrics.get("clinical_f1", 0.0) - rule.get("max_clinical_f1_drop", 0.03):
        return False
    if rule.get("reject_if_normal_collapse_increases", True):
        if optimized_metrics.get("normal_collapse_rate", 0.0) > baseline_metrics.get("normal_collapse_rate", 0.0):
            return False
    if rule.get("reject_if_new_hallucination_increases", True):
        if optimized_metrics.get("new_hallucination_cases_vs_baseline", 0) > baseline_metrics.get("new_hallucination_cases_vs_baseline", 0):
            return False
    return True


def save_style_profile(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_gate_report(report: dict, path: str | Path) -> None:
    """gate 결과를 JSON으로 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
