# 어디서부터 시작하면 되는가

이 프로젝트의 시작점은 이제 하나다.

```bash
python scripts/00_run_e2e_pipeline.py --config configs/pipeline/e2e_smoke.yaml
```

이 명령은 다음을 한 번에 실행한다.

1. Indiana CXR frontal/lateral paired sample 전처리
2. support/inference 7:3 multilabel stratified split
3. support set RAG DB 구축
4. TextGrad prompt optimization placeholder 실행
5. 6개 baseline/RAG ablation 실험 실행
6. final No-RAG vs Vision-RAG paired 실험 실행
7. summary/manifest 저장

## smoke와 full 차이

Smoke:

```bash
python scripts/00_run_e2e_pipeline.py --config configs/pipeline/e2e_smoke.yaml
```

- 목적: 전체 배선이 깨지지 않는지 확인
- inference sample: 12개만 사용
- output: `outputs/e2e_smoke`
- 현재 placeholder MedGemma/dummy_hash encoder 상태에서도 끝까지 돈다

Full:

```bash
python scripts/00_run_e2e_pipeline.py --config configs/pipeline/e2e_full.yaml
```

- 목적: 전체 inference set에 같은 pipeline 실행
- output: `outputs/e2e_full`
- production claim 전에는 실제 MedGemma generation, CheXbert labels, production retrieval encoder를 연결해야 한다

## 실행 후 어디를 보면 되는가

Smoke manifest:

```text
outputs/e2e_smoke/pipeline_manifest.json
```

전체 summary:

```text
outputs/e2e_smoke/summary/final_report.md
outputs/e2e_smoke/summary/main_results.csv
```

전처리 결과:

```text
outputs/e2e_smoke/preprocessed/indiana_paired_samples.csv
outputs/e2e_smoke/preprocessed/preprocess_report.md
```

split 품질:

```text
outputs/e2e_smoke/splits/split_summary.md
outputs/e2e_smoke/splits/split_quality_seed_0.csv
```

RAG DB:

```text
outputs/e2e_smoke/rag/db_summary.md
outputs/e2e_smoke/rag/support_metadata.csv 또는 support_metadata.parquet
outputs/e2e_smoke/rag/image_embeddings.npy
```

6개 실험:

```text
outputs/e2e_smoke/experiments/image_only_localization/predictions.csv
outputs/e2e_smoke/experiments/image_only_impression/predictions.csv
outputs/e2e_smoke/experiments/unrelated_rag_localization/predictions.csv
outputs/e2e_smoke/experiments/unrelated_rag_impression/predictions.csv
outputs/e2e_smoke/experiments/related_rag_localization/predictions.csv
outputs/e2e_smoke/experiments/related_rag_impression/predictions.csv
```

Final paired 비교:

```text
outputs/e2e_smoke/final_rag_vs_no_rag/predictions.csv
```

## 다음에 실제 논문/실험용으로 바꿀 순서

1. CheXbert label을 production으로 교체한다.
   - `scripts/11_export_chexbert_input.py`
   - `/root/CheXbert/run_chexbert.py`
   - `scripts/10_merge_chexbert_labels.py`

2. MedGemma generator placeholder를 실제 inference로 바꾼다.
   - `src/vision_rag_cxr/models/medgemma_generator.py`

3. dummy_hash retrieval encoder를 실제 encoder로 교체한다.
   - `configs/retrieval/hybrid_rag.yaml`
   - `src/vision_rag_cxr/models/medsiglip_encoder.py`
   - `src/vision_rag_cxr/models/medclip_encoder.py`
   - `src/vision_rag_cxr/models/biovil_encoder.py`
   - `src/vision_rag_cxr/models/chexzero_encoder.py`

4. TextGrad placeholder를 실제 optimization loop로 교체한다.
   - `src/vision_rag_cxr/experiments/prompt_optimization_experiment.py`
   - `src/vision_rag_cxr/prompting/textgrad_optimizer.py`

5. metric summary를 production metric으로 확장한다.
   - `src/vision_rag_cxr/evaluation/chexbert_metrics.py`
   - `src/vision_rag_cxr/evaluation/report_metrics.py`
   - `src/vision_rag_cxr/evaluation/localization_metrics.py`

## 기억할 점

Smoke가 통과했다는 뜻은 pipeline 구조와 파일 연결이 맞다는 뜻이다. 아직 scientific claim이 맞다는 뜻은 아니다. 현재 결과에는 placeholder MedGemma output과 dummy_hash retrieval encoder가 포함되어 있으므로, 실제 주장에는 production adapter 연결이 필요하다.
