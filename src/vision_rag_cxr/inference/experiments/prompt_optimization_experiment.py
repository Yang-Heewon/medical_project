"""TextGrad-style prompt optimization loop.

placeholder가 아니라 실제 iterative loop다. 단, MedGemma weight는 절대 수정하지 않고
STYLE_PROFILE prompt fragment만 critic 피드백으로 갱신한다.

loop (epoch마다):
1. critic(Qwen)이 baseline prediction vs GT를 보고 candidate STYLE_PROFILE을 제안한다.
2. MedGemma가 candidate prompt로 dev set impression을 재생성한다.
3. clinical metric(CheXbert micro-F1)과 per-sample 병변 점수를 계산한다.
4. lesion-preservation 통계 gate(non-inferiority/TOST)와 acceptance rule을 통과한
   candidate만 채택한다.

generator/critic backend가 placeholder여도 loop 배선과 gate는 그대로 실행되므로
smoke로 검증할 수 있고, backend를 transformers로 바꾸면 실제 최적화가 된다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vision_rag_cxr.datasets.labeler_chexbert import build_labeler
from vision_rag_cxr.evaluation.chexbert_metrics import multilabel_scores
from vision_rag_cxr.evaluation.report_metrics import compute_text_similarity_metrics
from vision_rag_cxr.models.critics.qwen import build_critic
from vision_rag_cxr.models.generators.factory import build_generator
from vision_rag_cxr.prompting.parser import parse_json_output
from vision_rag_cxr.prompting.prompt_templates import BASE_STYLE_PROFILE, IMPRESSION_PROMPT, render_prompt
from vision_rag_cxr.prompting.registry import build_style_profile
from vision_rag_cxr.prompting.textgrad_optimizer import (
    accept_optimized_prompt,
    evaluate_lesion_preservation_ttest,
    save_gate_report,
    save_style_profile,
)
from vision_rag_cxr.utils.io import ensure_dir, load_yaml

def _abnormal(labels: list[str]) -> list[str]:
    """label space에서 'No Finding'을 뺀 abnormal 라벨 목록 (데이터셋마다 다름)."""
    return [l for l in labels if l != "No Finding"]


def _gt_binary(row: dict, labels: list[str]) -> dict[str, int]:
    value = row.get("chexbert_labels_binary")
    if isinstance(value, str) and value.strip():
        d = json.loads(value)
    elif isinstance(value, dict):
        d = value
    else:
        d = {}
    return {label: int(d.get(label, 0)) for label in labels}


def _to_vec(d: dict[str, int], labels: list[str]) -> list[int]:
    return [int(d.get(l, 0)) for l in labels]


def _pred_binary_from_raw(raw: str, labeler) -> dict[str, int]:
    """생성된 impression(JSON or text)에서 예측 label dict를 만든다."""
    parsed, _ = parse_json_output(raw)
    if isinstance(parsed, dict):
        text = str(parsed.get("impression", "") or "")
        if parsed.get("mentioned_findings"):
            text += " " + " ".join(str(x) for x in parsed["mentioned_findings"])
    else:
        text = str(raw or "")
    binary, _ = labeler.label_report(text)
    return binary


def _set_f1(pred: dict[str, int], gt: dict[str, int], abn: list[str]) -> float:
    """positive label 집합 기준 per-sample F1. 둘 다 비면 1.0."""
    p = {l for l in abn if pred.get(l, 0) == 1}
    g = {l for l in abn if gt.get(l, 0) == 1}
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    inter = len(p & g)
    precision = inter / len(p)
    recall = inter / len(g)
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def _set_jaccard(a: dict[str, int], b: dict[str, int], abn: list[str]) -> float:
    pa = {l for l in abn if a.get(l, 0) == 1}
    pb = {l for l in abn if b.get(l, 0) == 1}
    if not pa and not pb:
        return 1.0
    union = len(pa | pb)
    return len(pa & pb) / union if union else 1.0


def _gen_one(generator, row: dict, style_profile: str, labeler) -> dict:
    # modality는 sample별로 채운다(데이터셋이 chest가 아닐 수 있음 → chest 고정 금지).
    prompt = render_prompt(IMPRESSION_PROMPT, style_profile=style_profile,
                           context_examples="", modality=row.get("modality"))
    raw = generator.generate_impression(row, prompt, context_examples=None)
    pred = _pred_binary_from_raw(raw, labeler)
    gt = _gt_binary(row, labeler.labels)
    abn = _abnormal(labeler.labels)
    parsed, _ = parse_json_output(raw)
    pred_text = str(parsed.get("impression", "")) if isinstance(parsed, dict) else str(raw or "")
    return {
        "uid": row.get("uid"),
        "raw": raw,
        "pred": pred,
        "gt": gt,
        "lesion_score": _set_f1(pred, gt, abn),    # 병변 정확도(제약용)
        "pred_text": pred_text,                     # 생성 impression 텍스트(목표용)
        "gt_text": str(row.get("impression", "") or ""),
    }


def _generate_dev(generators, dev_rows: list[dict], style_profile: str, labeler) -> list[dict]:
    """dev set impression 생성. generators가 여러 개면 GPU별 데이터-병렬(연속 chunk)로 동시 생성."""
    gens = generators if isinstance(generators, list) else [generators]
    if len(gens) == 1:
        return [_gen_one(gens[0], r, style_profile, labeler) for r in dev_rows]
    import math
    from concurrent.futures import ThreadPoolExecutor

    n = len(gens)
    size = math.ceil(len(dev_rows) / n)
    chunks = [dev_rows[i * size : (i + 1) * size] for i in range(n)]
    def _run(gi):
        return [_gen_one(gens[gi], r, style_profile, labeler) for r in chunks[gi]]
    with ThreadPoolExecutor(max_workers=n) as ex:
        parts = list(ex.map(_run, range(n)))  # 순서 보존(연속 chunk)
    out: list[dict] = []
    for p in parts:
        out.extend(p)
    return out


def _aggregate_metrics(results: list[dict], labels: list[str],
                       baseline_results: list[dict] | None = None) -> dict:
    """dev 결과에서 clinical/안전 metric을 집계한다 (label space는 labels로 주입)."""
    abn = _abnormal(labels)
    y_true = np.asarray([_to_vec(r["gt"], labels) for r in results], dtype=int)
    y_pred = np.asarray([_to_vec(r["pred"], labels) for r in results], dtype=int)
    metrics = multilabel_scores(y_true, y_pred)

    normal_collapse = halluc = omission = 0
    halluc_cases = 0
    for r in results:
        gt_abn = {l for l in abn if r["gt"].get(l, 0) == 1}
        pred_abn = {l for l in abn if r["pred"].get(l, 0) == 1}
        if gt_abn and not pred_abn:
            normal_collapse += 1
        new = pred_abn - gt_abn
        missed = gt_abn - pred_abn
        if new:
            halluc += 1
            halluc_cases += len(new)
        if missed:
            omission += 1
    n = max(1, len(results))
    # 텍스트(impression 스타일) 목표: 생성 impression vs 데이터셋 GT impression 유사도
    text_sim = {}
    if results and "pred_text" in results[0]:
        text_sim = compute_text_similarity_metrics(
            [r.get("pred_text", "") for r in results],
            [r.get("gt_text", "") for r in results],
        )
    # impression_style_score = BERTScore 우선, 없으면 ROUGE-L (최적화 objective)
    style_score = text_sim.get("bertscore_f1")
    if style_score is None:
        style_score = text_sim.get("rougeL_f")

    out = {
        "impression_style_score": float(style_score) if style_score is not None else 0.0,
        "impression_bertscore": text_sim.get("bertscore_f1"),
        "impression_rougeL": text_sim.get("rougeL_f"),
        "clinical_f1": metrics["chexbert_micro_f1"],
        "chexbert_macro_f1": metrics["chexbert_macro_f1"],
        "chexbert_micro_precision": metrics["chexbert_micro_precision"],
        "chexbert_micro_recall": metrics["chexbert_micro_recall"],
        "normal_collapse_rate": normal_collapse / n,
        "hallucination_rate": halluc / n,
        "omission_rate": omission / n,
        "new_hallucination_cases_vs_baseline": halluc_cases,
    }
    if baseline_results is not None:
        agree = np.mean([_set_jaccard(r["pred"], b["pred"], abn) for r, b in zip(results, baseline_results)])
        out["lesion_agreement_rate_vs_baseline"] = float(agree)
    return out


def _load_generators_critic(config: dict):
    """generator(들) + critic 로드. config['generator_devices']가 있으면 GPU별 replica를 만든다."""
    gen_cfg = config.get("generator_config")
    gen_cfg = load_yaml(gen_cfg) if isinstance(gen_cfg, str) else (gen_cfg or {})
    crit_cfg = config.get("critic_config")
    crit_cfg = load_yaml(crit_cfg) if isinstance(crit_cfg, str) else (crit_cfg or {})

    gdevs = config.get("generator_devices")  # 예: ["cuda:0","cuda:1"] (visible 기준)
    if gdevs:
        generators = []
        for d in gdevs:
            gc = dict(gen_cfg); gc["device"] = "cuda"; gc["device_map"] = d
            generators.append(build_generator(gc))
    else:
        generators = [build_generator(gen_cfg)]
    # critic이 데이터셋 label space(어떤 finding을 보고하는지)를 '인지'하게 주입한다.
    # → critique/rewrite가 그 데이터셋의 finding 어휘에 맞춰 STYLE_PROFILE을 최적화.
    label_space = config.get("label_space", "chexbert_14")
    labeler = build_labeler({"label_space": label_space})
    crit_cfg.setdefault("label_space", label_space)
    crit_cfg.setdefault("label_names", _abnormal(labeler.labels))
    # 중요: transformers from_pretrained는 스레드 동시 로딩에 안전하지 않다.
    # 생성기들을 '순차로' 먼저 로드해 두면, 이후 _generate_dev의 스레드 병렬 생성은 안전하다.
    for g in generators:
        try:
            g.load()
        except Exception as e:
            print(f"[textgrad] generator load warn: {e}", flush=True)
    return generators, build_critic(crit_cfg), labeler


def _trace_init(path: Path, config: dict, init_style: str, n_dev: int, n_gen: int) -> None:
    gate = config.get("lesion_preservation_gate", {}) or {}
    lines = [
        "# TextGrad 진행 기록 (prompt optimization trace)",
        "",
        "VLM은 frozen — 프롬프트(STYLE_PROFILE)만 바꿔 impression 텍스트를 개선하되, 병변 검출은 baseline과 동일 유지.",
        "",
        f"- **objective**: `{config.get('objective_metric','impression_style_score')}` (impression 텍스트 스타일: BERTScore/ROUGE vs GT impression)",
        f"- **constraint(병변)**: lesion-preservation gate `mode={gate.get('mode','noninferiority')}` margin={gate.get('margin',0.03)} alpha={gate.get('alpha',0.05)} + collapse/hallucination 증가 금지",
        f"- dev_samples: {n_dev} | generator GPU replica: {n_gen} | max_epochs: {config.get('max_epochs')} | patience: {config.get('early_stop_patience')}",
        "",
        "## epoch 0 — baseline STYLE_PROFILE",
        "```text",
        init_style.strip(),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _trace_epoch(path: Path, epoch: int, adopt: bool, cand_metrics: dict, best_metrics: dict,
                 gate: dict, critiques: list[str], candidate_style: str, objective_metric: str) -> None:
    verdict = "✅ ACCEPTED" if adopt else "❌ REJECTED"
    blk = [
        f"## epoch {epoch} — {verdict}",
        f"- {objective_metric}: **{round(cand_metrics.get('impression_style_score',0.0),4)}** "
        f"(best {round(best_metrics.get('impression_style_score',0.0),4)}) | "
        f"BERTScore={cand_metrics.get('impression_bertscore')} ROUGE-L={cand_metrics.get('impression_rougeL')}",
        f"- clinical_f1={round(cand_metrics.get('clinical_f1',0.0),4)} | normal_collapse={round(cand_metrics.get('normal_collapse_rate',0.0),4)} | hallucination={round(cand_metrics.get('hallucination_rate',0.0),4)}",
        f"- 병변 게이트: pass={gate.get('lesion_preservation_pass')} reason={gate.get('reason')} "
        f"mean_delta={gate.get('mean_delta_candidate_minus_baseline')} (mode={gate.get('mode')})",
        "",
        "### critic 피드백 (현재 best 대비, 일부)",
    ]
    for c in critiques[:3]:
        blk.append(f"- {str(c).strip()[:400]}")
    blk += [
        "",
        "### critic이 제안한 candidate STYLE_PROFILE",
        "```text",
        str(candidate_style).strip(),
        "```",
        "",
    ]
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(blk) + "\n")


def run_prompt_optimization(config: dict) -> dict:
    """STYLE_PROFILE을 critic 피드백으로 최적화하고 최선 prompt를 저장한다."""
    out_dir = ensure_dir(config.get("output_dir", "outputs/prompt_optimization"))
    max_epochs = int(config.get("max_epochs", 4))
    dev_size = int(config.get("dev_sample_size", 64))
    critique_size = int(config.get("critique_sample_size", min(8, dev_size)))
    # style_profile_init은 plug-in: 카탈로그 이름 | 파일 경로 | 리터럴 텍스트 모두 허용.
    init_style = build_style_profile(config.get("style_profile_init"), default=BASE_STYLE_PROFILE)
    accept_rule = config.get("acceptance_rule", {})
    gate_cfg = config.get("lesion_preservation_gate", {}) or {}

    # dev set 로드 (split의 inference에서 dev subset을 떼어 쓴다).
    split_csv = config.get("split_csv")
    if not split_csv or not Path(split_csv).exists():
        # split이 없으면 최적화를 실행할 수 없으므로 init을 그대로 저장하고 알린다.
        save_style_profile(init_style, Path(out_dir) / "optimized_style_profile.txt")
        report = {"status": "no_split_csv", "reason": f"split_csv not found: {split_csv}"}
        save_gate_report(report, Path(out_dir) / "lesion_preservation_gate.json")
        (Path(out_dir) / "prompt_optimization_summary.md").write_text(
            "# Prompt optimization\n\nsplit_csv가 없어 최적화를 건너뛰고 init STYLE_PROFILE을 저장했습니다.\n",
            encoding="utf-8",
        )
        return report

    df = pd.read_csv(split_csv)
    dev = df[df["split"] == "inference"].head(dev_size)
    if len(dev) == 0:
        dev = df.head(dev_size)
    dev_rows = [r.to_dict() for _, r in dev.iterrows()]

    generators, critic, labeler = _load_generators_critic(config)
    abn = _abnormal(labeler.labels)                    # 이 데이터셋의 abnormal finding 목록
    trace_path = Path(out_dir) / "textgrad_trace.md"   # 프롬프트/critic 진행 상세 기록
    _trace_init(trace_path, config, init_style, len(dev_rows), len(generators))

    # baseline
    baseline_results = _generate_dev(generators, dev_rows, init_style, labeler)
    baseline_metrics = _aggregate_metrics(baseline_results, labeler.labels, baseline_results)
    baseline_lesion = [r["lesion_score"] for r in baseline_results]
    _trace_append = lambda t: open(trace_path, "a", encoding="utf-8").write(t + "\n")
    _trace_append(
        f"- baseline: impression_style_score={round(baseline_metrics.get('impression_style_score',0.0),4)} "
        f"(BERTScore={baseline_metrics.get('impression_bertscore')}, ROUGE-L={baseline_metrics.get('impression_rougeL')}), "
        f"clinical_f1={round(baseline_metrics['clinical_f1'],4)}, normal_collapse={round(baseline_metrics['normal_collapse_rate'],4)}\n"
    )

    best_style = init_style
    best_metrics = baseline_metrics
    best_gate = {"reason": "baseline", "lesion_preservation_pass": True}
    history = [
        {
            "epoch": 0,
            "stage": "baseline",
            "impression_style_score": round(baseline_metrics.get("impression_style_score", 0.0), 4),
            "clinical_f1": round(baseline_metrics["clinical_f1"], 4),
            "normal_collapse_rate": round(baseline_metrics["normal_collapse_rate"], 4),
            "accepted": True,
        }
    ]

    current_style = init_style
    current_results = baseline_results       # 현재 best의 per-sample 결과 (반복 탐색의 기준)
    # 최적화 목표: 기본은 impression 텍스트 스타일 점수(BERTScore/ROUGE). 병변정확도는 제약(게이트).
    objective_metric = config.get("objective_metric", "impression_style_score")
    patience = int(config.get("early_stop_patience", 10))
    no_improve = 0
    hist_path = Path(out_dir) / "prompt_optimization_history.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)  # baseline 즉시 기록

    for epoch in range(1, max_epochs + 1):
        # 1. critic 피드백: '현재 best' 예측 기준으로 비평 -> epoch마다 새 방향 탐색
        critiques = []
        for r in current_results[:critique_size]:
            gt_abn = {l for l in abn if r["gt"].get(l, 0) == 1}
            pred_abn = {l for l in abn if r["pred"].get(l, 0) == 1}
            sample_metrics = {
                "missed_labels": sorted(gt_abn - pred_abn),
                "hallucinated_labels": sorted(pred_abn - gt_abn),
                "normal_collapse": bool(gt_abn and not pred_abn),
            }
            # 참조(target)로 실제 GT impression을, prediction으로 생성 impression 텍스트를 넘긴다(버그 수정).
            critiques.append(
                critic.critique(str(r.get("pred_text", r["raw"])), str(r.get("gt_text", "")), r, sample_metrics)
            )

        # 2. candidate STYLE_PROFILE 생성 (현재 best prompt를 critic이 개선)
        candidate_style = critic.rewrite_style_profile(current_style, critiques, best_metrics)

        # 3. candidate로 재생성 + metric
        cand_results = _generate_dev(generators, dev_rows, candidate_style, labeler)
        cand_metrics = _aggregate_metrics(cand_results, labeler.labels, baseline_results)
        cand_lesion = [r["lesion_score"] for r in cand_results]

        # 4. 병변 보존 통계 gate (병변 유사도 유지 검정)
        gate = evaluate_lesion_preservation_ttest(
            baseline_lesion, cand_lesion,
            margin=float(gate_cfg.get("margin", 0.03)),
            alpha=float(gate_cfg.get("alpha", 0.05)),
            mode=gate_cfg.get("mode", "noninferiority"),
            min_samples=int(gate_cfg.get("min_samples", 30)),
        )
        cand_metrics["lesion_preservation_gate"] = gate

        accepted = accept_optimized_prompt(baseline_metrics, cand_metrics, accept_rule)
        # 채택 = 병변 정확도 제약(게이트/규칙) 통과 AND impression 스타일 목표가 개선
        improved = cand_metrics.get(objective_metric, 0.0) >= best_metrics.get(objective_metric, 0.0) - 1e-9
        adopt = accepted and improved
        if adopt:
            best_style = candidate_style
            best_metrics = cand_metrics
            best_gate = gate
            current_style = candidate_style
            current_results = cand_results
            no_improve = 0
        else:
            no_improve += 1

        history.append(
            {
                "epoch": epoch,
                "stage": "candidate",
                "impression_style_score": round(cand_metrics.get("impression_style_score", 0.0), 4),
                "best_style_score": round(best_metrics.get("impression_style_score", 0.0), 4),
                "clinical_f1": round(cand_metrics["clinical_f1"], 4),
                "normal_collapse_rate": round(cand_metrics["normal_collapse_rate"], 4),
                "lesion_gate_pass": bool(gate.get("lesion_preservation_pass")),
                "accepted": bool(adopt),
                "no_improve": no_improve,
            }
        )
        pd.DataFrame(history).to_csv(hist_path, index=False)  # epoch별 진행 즉시 저장(모니터링용)
        save_style_profile(best_style, Path(out_dir) / "optimized_style_profile.txt")
        _trace_epoch(trace_path, epoch, adopt, cand_metrics, best_metrics, gate, critiques, candidate_style, objective_metric)

        # early stopping: patience epoch 동안 개선 없으면 종료
        if no_improve >= patience:
            history.append({"epoch": epoch, "stage": f"early_stop(patience={patience})", "accepted": False})
            pd.DataFrame(history).to_csv(hist_path, index=False)
            break

    # 저장
    save_style_profile(best_style, Path(out_dir) / "optimized_style_profile.txt")
    save_gate_report(best_gate, Path(out_dir) / "lesion_preservation_gate.json")
    pd.DataFrame(history).to_csv(Path(out_dir) / "prompt_optimization_history.csv", index=False)

    summary = [
        "# Prompt optimization summary",
        "",
        f"- dev_samples: {len(dev_rows)}",
        f"- max_epochs: {max_epochs}",
        f"- generator_backend: {getattr(generators[0], 'backend', 'unknown')}",
        f"- critic_backend: {getattr(critic, 'backend', 'unknown')}",
        f"- baseline_clinical_f1: {round(baseline_metrics['clinical_f1'], 4)}",
        f"- best_clinical_f1: {round(best_metrics['clinical_f1'], 4)}",
        f"- adopted_epochs: {[h['epoch'] for h in history if h.get('accepted') and h['epoch'] > 0]}",
        "",
        "## Final lesion preservation gate",
        "```json",
        json.dumps(best_gate, indent=2, ensure_ascii=False),
        "```",
        "",
        "Note: generator/critic가 placeholder면 결과는 배선 검증용이다. "
        "실제 최적화에는 backend: transformers와 모델 weight가 필요하다.",
    ]
    (Path(out_dir) / "prompt_optimization_summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {"best_clinical_f1": best_metrics["clinical_f1"], "history": history, "gate": best_gate}
