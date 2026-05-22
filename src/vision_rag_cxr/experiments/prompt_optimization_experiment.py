"""TextGrad prompt optimization experiment skeleton.

현재 파일은 production TextGrad loop가 들어갈 위치를 고정한다.
실제 최적화에서는 Qwen critic이 candidate STYLE_PROFILE을 만들고,
병변 보존 gate가 통과한 candidate만 최종 prompt로 저장한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vision_rag_cxr.prompting.textgrad_optimizer import (
    evaluate_lesion_preservation_ttest,
    save_gate_report,
    save_style_profile,
)
from vision_rag_cxr.utils.io import ensure_dir


def _maybe_evaluate_lesion_gate(config: dict) -> dict:
    """이미 계산된 baseline/candidate lesion score CSV가 있으면 통계 gate를 실행한다.

    production loop에서는 candidate prompt를 만들 때마다 같은 dev uid에 대해
    baseline MedGemma localization score와 candidate localization score를 저장한 뒤 이 gate를 호출한다.
    """
    gate_cfg = config.get("lesion_preservation_gate", {}) or {}
    if not gate_cfg.get("enabled", False):
        return {"enabled": False, "reason": "disabled"}

    score_csv = gate_cfg.get("score_csv")
    if not score_csv or not Path(score_csv).exists():
        return {
            "enabled": True,
            "lesion_preservation_pass": False,
            "reason": "score_csv_missing_placeholder_not_production",
            "expected_score_csv": score_csv,
            "expected_columns": [
                gate_cfg.get("baseline_score_column", "baseline_lesion_score"),
                gate_cfg.get("candidate_score_column", "candidate_lesion_score"),
            ],
        }

    df = pd.read_csv(score_csv)
    baseline_col = gate_cfg.get("baseline_score_column", "baseline_lesion_score")
    candidate_col = gate_cfg.get("candidate_score_column", "candidate_lesion_score")
    missing = [c for c in [baseline_col, candidate_col] if c not in df.columns]
    if missing:
        return {"enabled": True, "lesion_preservation_pass": False, "reason": f"missing_columns: {missing}"}

    return evaluate_lesion_preservation_ttest(
        df[baseline_col].tolist(),
        df[candidate_col].tolist(),
        margin=float(gate_cfg.get("margin", 0.03)),
        alpha=float(gate_cfg.get("alpha", 0.05)),
        mode=gate_cfg.get("mode", "noninferiority"),
        min_samples=int(gate_cfg.get("min_samples", 30)),
    )


def run_prompt_optimization(config: dict) -> None:
    """STYLE_PROFILE 최적화 placeholder.

    현재는 실제 TextGrad 호출 대신 저장/게이트 구조를 만든다.
    production에서는 이 함수 안에서 다음 loop를 수행한다.
    1. Qwen critic이 candidate STYLE_PROFILE 생성
    2. MedGemma가 candidate prompt로 impression 생성
    3. clinical metric 계산
    4. 같은 dev uid에서 baseline/candidate localization score 계산
    5. lesion-preservation t-test gate 통과 시 candidate 채택
    """
    out_dir = ensure_dir(config.get("output_dir", "outputs/prompt_optimization"))
    init = config.get("style_profile_init", "")
    optimized = init + "\nWhen uncertain, describe uncertainty explicitly rather than forcing a normal impression."

    lesion_gate_report = _maybe_evaluate_lesion_gate(config)
    save_gate_report(lesion_gate_report, Path(out_dir) / "lesion_preservation_gate.json")

    # placeholder 단계에서는 gate가 통과하지 않아도 파일을 저장한다.
    # production 단계에서는 accept_optimized_prompt()로 최종 채택 여부를 결정해야 한다.
    save_style_profile(optimized, Path(out_dir) / "optimized_style_profile.txt")
    summary = [
        "# Prompt optimization summary",
        "",
        "Placeholder optimizer completed. Connect TextGrad objective for production.",
        "",
        "## Lesion preservation gate",
        "```json",
        json.dumps(lesion_gate_report, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Intended production loop",
        "1. Generate candidate STYLE_PROFILE with Qwen critic.",
        "2. Evaluate impression quality against GT impression.",
        "3. Evaluate baseline vs candidate lesion localization scores on the same dev uid set.",
        "4. Accept only if clinical metric gate and lesion-preservation statistical gate both pass.",
    ]
    (Path(out_dir) / "prompt_optimization_summary.md").write_text("\n".join(summary), encoding="utf-8")
