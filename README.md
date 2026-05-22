# vision-rag-cxr

Indiana University Chest X-ray 기반 Vision-RAG / Prompt Optimization 실험 프레임워크입니다.

## 연구 설계 요약

이 프로젝트는 다음 질문을 검증합니다.

1. `Image only`보다 `related image+impression RAG`가 병변 탐지 및 impression 생성에 도움이 되는가?
2. `unrelated RAG`와 `related RAG`의 차이를 통해 retrieval relevance 효과를 보일 수 있는가?
3. TextGrad 기반 prompt optimization이 report style/clinical agreement를 높이면서 lesion detection ability를 해치지 않는가?
4. MedGemma generator와 MedSigLIP/MedCLIP/BioViL/CheXzero retrieval encoder를 분리해서 plug-and-play로 교체할 수 있는가?

## Labeling policy

기본 정책:

```text
primary_label_source = impression
secondary_label_source = findings + impression
auxiliary_audit_source = MeSH + Problems
label_space = CheXpert/CheXbert 14 labels
uncertain_policy = U-Ones
save_raw_uncertain = true
```

해석:
- `impression`은 radiology report의 결론 section이라 evaluation target과 가장 직접적으로 맞습니다.
- `findings + impression`은 impression이 비어 있거나 너무 짧은 경우 secondary source로 사용합니다.
- `MeSH + Problems`는 CheXbert 입력이 아니라 auxiliary audit source입니다.
- main label은 multi-label로 유지합니다. top-1 label은 사용하지 않습니다.
- uncertain은 기본적으로 U-Ones로 처리하지만 raw uncertain label도 저장합니다.

## Human audit mode

Indiana IU에는 기본 bbox GT가 없으므로 strict bbox mAP를 주장하지 않습니다.  
대신 inference set 일부를 샘플링하여 사람이 다음 CSV 형식으로 box correctness를 검수합니다.

```csv
uid,experiment_name,lesion_id,model_label,model_anatomy,model_bbox,human_correct,human_notes
```

이 결과는 `src/vision_rag_cxr/evaluation/localization_metrics.py`의 `summarize_human_audit()`로 요약합니다.

## 데이터 경로

서버 기준:

```text
/root/data/images/images_normalized
/root/data/label/indiana_reports.csv
```


## One-command pipeline

처음 시작할 때는 아래 smoke부터 실행하세요. 전체 단계가 한 번에 돌아가며 산출물은 `outputs/e2e_smoke`에 저장됩니다.

```bash
python scripts/00_run_e2e_pipeline.py --config configs/pipeline/e2e_smoke.yaml
```

전체 inference set으로 같은 구조를 돌릴 때는 다음을 사용합니다.

```bash
python scripts/00_run_e2e_pipeline.py --config configs/pipeline/e2e_full.yaml
```

시작 가이드는 `docs/START_HERE_KO.md`, 전체 연구 설계/prompt는 `docs/PROJECT_FRAMEWORK_PROMPT_KO.md`, TextGrad 병변 보존 gate는 `docs/TEXTGRAD_LESION_PRESERVATION_KO.md`를 보세요.

## One-click server check

다른 서버에서 1차 검증은 아래 명령 하나로 실행할 수 있습니다.

```bash
RUN_NAME=first_check \
GPU_IDS=1,2,3,4,6 \
SAMPLE_SIZE=8 \
bash scripts/run_one_click_smoke_and_prompt_check.sh
```

이 명령은 빠른 E2E smoke, CheXbert production label 확인/생성, production split, MedGemma prompt-preservation sampling t-test, HTML audit 생성을 순서대로 수행합니다.

자세한 새 서버 설정은 `docs/SERVER_SETUP_KO.md`를 보세요.

## 실행 순서

### 0. Docker build

```bash
cd /workspace/vision-rag-cxr
docker build -f docker/Dockerfile -t vision-rag-cxr:latest .
docker compose -f docker/docker-compose.gpu.yaml run --rm vision-rag-cxr
```

### 1. 전처리

```bash
python scripts/01_preprocess_indiana.py --config configs/data/indiana.yaml
```

출력:

```text
/workspace/outputs/preprocessed/indiana_paired_samples.csv
/workspace/outputs/preprocessed/indiana_paired_samples.jsonl
/workspace/outputs/preprocessed/label_distribution.csv
/workspace/outputs/preprocessed/preprocess_report.md
```

### 2. Support / inference split

```bash
python scripts/02_create_stratified_split.py \
  --data_csv /workspace/outputs/preprocessed/indiana_paired_samples.csv \
  --config configs/split/stratified_70_30.yaml
```

출력:

```text
/workspace/outputs/splits/split_seed_0.csv
/workspace/outputs/splits/split_quality_seed_0.csv
/workspace/outputs/splits/split_summary.md
```

### 3. Support DB 구축

```bash
python scripts/03_build_support_database.py \
  --split_csv /workspace/outputs/splits/split_seed_0.csv \
  --config configs/retrieval/hybrid_rag.yaml
```

출력:

```text
/workspace/outputs/rag/faiss_index/support.index
/workspace/outputs/rag/support_metadata.parquet
/workspace/outputs/rag/image_embeddings.npy
/workspace/outputs/rag/text_embeddings.npy
/workspace/outputs/rag/label_vectors.npy
/workspace/outputs/rag/db_summary.md
```

### 4. Baseline experiments

```bash
python scripts/04_run_baseline_experiments.py \
  --config configs/experiments/exp_01_image_only_localization.yaml

python scripts/04_run_baseline_experiments.py \
  --config configs/experiments/exp_02_image_only_impression.yaml
```

### 5. TextGrad prompt optimization

```bash
python scripts/05_run_textgrad_prompt_optimization.py \
  --config configs/experiments/textgrad_prompt_optimization.yaml
```

### 6. RAG ablation 6개 실험

```bash
python scripts/06_run_rag_ablation.py --config configs/experiments/exp_03_unrelated_rag_localization.yaml
python scripts/06_run_rag_ablation.py --config configs/experiments/exp_04_unrelated_rag_impression.yaml
python scripts/06_run_rag_ablation.py --config configs/experiments/exp_05_related_rag_localization.yaml
python scripts/06_run_rag_ablation.py --config configs/experiments/exp_06_related_rag_impression.yaml
```

### 7. Final RAG vs No-RAG

```bash
python scripts/07_run_final_rag_experiment.py \
  --config configs/experiments/final_rag_vs_no_rag.yaml
```

### 8. 결과 요약

```bash
python scripts/08_summarize_results.py \
  --results_root /workspace/outputs/experiments \
  --out_dir /workspace/outputs/summary
```

## 수정 위치

| 바꾸고 싶은 것 | 수정 파일 |
|---|---|
| 데이터 경로 | `configs/data/indiana.yaml` |
| labeling policy | `configs/data/indiana.yaml` |
| split 비율/seed | `configs/split/stratified_70_30.yaml` |
| MedGemma 설정 | `configs/models/medgemma.yaml` |
| Qwen critic 설정 | `configs/models/qwen35_9b_critic.yaml` |
| retrieval encoder | `configs/retrieval/hybrid_rag.yaml` |
| MedSigLIP 구현 | `src/vision_rag_cxr/models/medsiglip_encoder.py` |
| MedCLIP 구현 | `src/vision_rag_cxr/models/medclip_encoder.py` |
| BioViL 구현 | `src/vision_rag_cxr/models/biovil_encoder.py` |
| CheXzero 구현 | `src/vision_rag_cxr/models/chexzero_encoder.py` |
| prompt template | `src/vision_rag_cxr/prompting/prompt_templates.py` |
| TextGrad objective | `src/vision_rag_cxr/prompting/textgrad_optimizer.py` |
| human audit metric | `src/vision_rag_cxr/evaluation/localization_metrics.py` |


## Current implementation status

상세 연구 설계, 의사결정, plug-and-play 수정 위치, 다음 작업용 master prompt는 `docs/PROJECT_FRAMEWORK_PROMPT_KO.md`에 정리되어 있습니다.

현재 검증된 부분:
- Indiana 전처리: 실제 `/root/data` 기준 3,405 paired uid 생성 확인
- CheXbert external inference 결과를 canonical paired CSV에 병합하는 경로 확인
- 7:3 multilabel stratified split 생성 확인
- dummy_hash 기반 빠른 E2E smoke 확인
- MedGemma HF/transformers real generation adapter 연결
- BiomedCLIP/OpenCLIP 기반 image/text encoder adapter 연결
- prompt-preservation sampling t-test 및 HTML audit 생성 확인
- one-click smoke runner: `scripts/run_one_click_smoke_and_prompt_check.sh`

아직 production claim 전에 더 정리해야 하는 부분:
- 3~6번 ablation의 `related/unrelated` k-shot selection을 MeSH/Problems controlled DB 기준으로 교체해야 합니다.
- TextGrad optimization loop는 gate/acceptance scaffold가 있고, 실제 iterative critic rewrite loop는 추가 연결이 필요합니다.
- IU-only에서는 공식 bbox GT가 없으므로 bbox 정답성은 HTML human audit 또는 별도 pseudo/manual GT가 필요합니다.
- MedSigLIP/BioViL/CheXzero adapters는 skeleton이며, 현재 실제로 검증된 retrieval adapter는 BiomedCLIP/OpenCLIP 계열입니다.

## 주의사항

- 이 template의 기본 encoder는 `dummy_hash`입니다. pipeline 검증용이며 논문 결과에 쓰면 안 됩니다.
- 실제 DB 구축 실험에서는 `configs/retrieval/hybrid_rag.yaml`의 `vision_encoder_name`을 `medsiglip`, `medclip`, `biovil`, `chexzero` 중 하나로 바꾸고 해당 adapter를 구현해야 합니다.
- `keyword_fallback` labeler는 CheXbert 설치 전 pipeline 테스트용입니다. 실제 결과는 CheXbert backend로 생성해야 합니다.
- IU-only에서는 bbox GT가 없으므로 human audit mode를 사용하세요.
