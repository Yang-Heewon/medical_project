# Vision-RAG-CXR E2E Framework Prompt / 설계 문서

이 문서는 Indiana University Chest X-ray 기반 Vision-RAG, VLM 병변 탐지, TextGrad prompt optimization, final RAG-vs-NoRAG 실험을 하나의 모듈형 프레임워크로 구현/확장하기 위한 master prompt이자 설계 기준이다.

## 1. 연구 목표

목표는 `Image only` VLM baseline과 `Vision-RAG k-shot examples`를 넣은 VLM을 같은 inference set에서 비교하여, 관련 support example이 impression 생성과 병변 localization을 개선하는지 검증하는 것이다.

핵심 질문은 다음과 같다.

1. MedGemma 같은 VLM이 CXR image pair만 보고 병변을 제대로 언급/탐지하는가?
2. unrelated support example은 단순 prompt length/few-shot 효과만 주는 negative control로 볼 수 있는가?
3. related support example은 disease/anatomy/visual similarity가 맞는 사례를 제공하여 impression과 localization을 개선하는가?
4. TextGrad는 VLM weight를 바꾸지 않고 `STYLE_PROFILE` prompt fragment만 최적화하면서, 병변 보존 능력을 해치지 않는가?
5. retrieval encoder를 MedSigLIP, MedCLIP, BioViL, CheXzero 등으로 교체했을 때 final Vision-RAG 성능이 어떻게 달라지는가?

## 2. 데이터 정책

원본 경로는 서버 기준 다음과 같다.

- image root: `/root/data/images/images_normalized`
- report CSV: `/root/data/label/indiana_reports.csv`
- expected columns: `uid,MeSH,Problems,image,indication,comparison,findings,impression`

전처리 정책:

- split unit은 반드시 `uid`이다.
- frontal + lateral image가 모두 있는 uid만 사용한다.
- `indiana_projections.csv` 같은 projection metadata가 있으면 우선 사용한다.
- projection metadata가 없으면 uid prefix image list를 정렬해 fallback pair를 만든다.
- fallback projection은 report에 warning으로 남긴다.
- raw data는 절대 덮어쓰지 않고 `outputs/preprocessed` 아래에 canonical table을 저장한다.

현재 코드 위치:

- 전처리 entrypoint: `scripts/01_preprocess_indiana.py`
- pair matcher: `src/vision_rag_cxr/data/projection_pair_matcher.py`
- canonical preprocessor: `src/vision_rag_cxr/data/indiana_preprocessor.py`
- config: `configs/data/indiana.yaml`

## 3. Labeling 정책

기본 label space는 CheXpert/CheXbert 14 labels이다.

권장 정책:

- primary label source: `impression`
- secondary label source: `findings + impression`
- auxiliary audit source: `MeSH + Problems`
- main task label type: multi-label
- uncertain policy: U-Ones, 단 raw uncertain label은 따로 보존

의사결정:

- top-1 label로 줄이지 않는다. CXR report는 cardiomegaly + effusion + atelectasis처럼 여러 병변이 동시에 존재할 수 있으므로 top-1은 clinically lossy하다.
- MeSH/Problems를 primary GT로 쓰지 않는다. dataset annotation 성격이 강하고 impression target과 불일치할 수 있어 audit feature로 둔다.
- CheXbert label을 impression 평가의 clinical agreement metric으로 사용한다.
- fallback keyword labeler는 smoke test 전용이다. 논문/보고 결과에는 CheXbert 실행 결과를 merge한 CSV를 사용한다.

현재 코드 위치:

- fallback/CheXbert adapter 위치: `src/vision_rag_cxr/data/labeler_chexbert.py`
- CheXbert input export: `scripts/11_export_chexbert_input.py`
- CheXbert output merge: `scripts/10_merge_chexbert_labels.py`
- split 결과 예시: `outputs/splits_chexbert`

## 4. Support / Inference Split

기본 split은 support:inference = 7:3이다.

권장 방식:

- multilabel iterative stratification을 사용한다.
- seed `[0,1,2,3]`를 저장하여 robustness를 본다.
- label distribution drift를 `split_quality_seed_*.csv`로 저장한다.
- 동일 uid가 support와 inference에 동시에 들어가면 안 된다.

질병 분포를 맞추기 위해 Gaussian sampling을 별도로 설계할 필요는 낮다. 실제 label vector가 binary multi-label이므로 Gaussian 가정은 부자연스럽고, iterative stratification이 더 직접적이다. 다만 rare disease가 너무 적으면 abnormal-heavy subset을 별도 analysis set으로 추가할 수 있다.

현재 코드 위치:

- split entrypoint: `scripts/02_create_stratified_split.py`
- split module: `src/vision_rag_cxr/data/splitters.py`
- config: `configs/split/stratified_70_30.yaml`

## 5. RAG DB 설계

support set만 사용해서 database를 만든다.

저장 key:

- image embedding
- text/report embedding
- CheXbert label vector
- anatomy/pathology phrase
- metadata: uid, frontal/lateral path, impression, findings, MeSH, Problems, labels

중요한 leakage 정책:

- final/no-leakage retrieval은 query image embedding만 사용한다.
- query GT CheXbert label을 retrieval에 쓰면 oracle retrieval이므로 final claim에 쓰지 않는다.
- `query_label_source: chexbert_labels_binary`는 debugging/upper-bound 실험으로만 쓴다.

현재 코드 위치:

- DB build: `scripts/03_build_support_database.py`
- DB module: `src/vision_rag_cxr/rag/build_database.py`
- related retriever: `src/vision_rag_cxr/rag/related_retriever.py`
- unrelated retriever: `src/vision_rag_cxr/rag/unrelated_retriever.py`
- retriever factory: `src/vision_rag_cxr/rag/retriever_factory.py`
- context formatter: `src/vision_rag_cxr/rag/prompt_context_builder.py`
- config: `configs/retrieval/hybrid_rag.yaml`

## 6. 실험 세트

필수 6개 실험:

1. `image_only_localization`: image만 넣고 lesion bbox JSON 생성
2. `image_only_impression`: image만 넣고 impression JSON 생성
3. `unrelated_rag_localization`: unrelated support examples + image로 bbox 생성
4. `unrelated_rag_impression`: unrelated support examples + image로 impression 생성
5. `related_rag_localization`: related support examples + image로 bbox 생성
6. `related_rag_impression`: related support examples + image로 impression 생성

비교:

- localization: 1 vs 3 vs 5
- impression: 2 vs 4 vs 6

해석:

- `5 > 1`이면 related visual/text examples가 localization에 도움을 준다.
- `6 > 2`이면 related examples가 impression generation에 도움을 준다.
- `3`/`4`는 unrelated few-shot control이다. `unrelated > image_only`일 수 있지만, `related > unrelated`가 핵심 claim이다.

현재 코드 위치:

- baseline runner: `scripts/04_run_baseline_experiments.py`
- RAG runner: `scripts/06_run_rag_ablation.py`
- localization class: `src/vision_rag_cxr/experiments/localization_experiment.py`
- impression class: `src/vision_rag_cxr/experiments/impression_experiment.py`
- RAG wrapper: `src/vision_rag_cxr/experiments/rag_ablation_experiment.py`

## 7. Localization metric 정책

Indiana IU-only에는 기본 bbox GT가 없다. 따라서 strict mAP/IoU를 최종 localization 성능으로 주장하면 안 된다.

권장 평가:

- model output bbox를 export한다.
- human audit sheet에서 사람이 lesion label/anatomy/bbox correctness를 평가한다.
- metric은 box correctness rate, per-label correctness, failure mode summary를 쓴다.
- 별도 pseudo/manual bbox subset이 있으면 그 subset에 한해 IoU/label match를 보조 metric으로 제시한다.

현재 코드 위치:

- metric: `src/vision_rag_cxr/evaluation/localization_metrics.py`
- audit export 후보: `scripts/09_create_human_audit_sheet.py`

## 8. Impression metric 정책

권장 metric 묶음:

- CheXbert/CheXpert 14-label micro/macro F1, precision, recall
- per-label F1 for clinically important labels
- BERTScore/ROUGE 같은 natural language similarity
- hallucination rate: GT에 없는 abnormal label을 새로 만든 비율
- omission rate: GT positive label을 누락한 비율
- normal collapse rate: abnormal GT인데 no finding/normal로 생성한 비율

통계:

- 같은 inference uid에 대한 paired comparison으로 paired t-test를 기본 제공한다.
- 점수 분포가 비정규/희귀 label 중심이면 Wilcoxon signed-rank와 bootstrap CI를 같이 보고한다.
- t-test는 4회 반복 실험의 aggregate만 보는 것보다 uid-level paired score를 seed별로 보고 결합하는 방식이 더 안전하다.

현재 코드 위치:

- CheXbert metrics: `src/vision_rag_cxr/evaluation/chexbert_metrics.py`
- text metrics: `src/vision_rag_cxr/evaluation/report_metrics.py`
- stats: `src/vision_rag_cxr/evaluation/statistical_tests.py`

## 9. TextGrad Prompt Optimization 설계

TextGrad의 목적은 model weight update가 아니라 prompt variable update이다.

권장 variable:

- `STYLE_PROFILE`만 optimize한다.
- task schema, output JSON schema, safety constraints는 고정한다.

목표:

- Image + prompt -> generated impression
- generated impression이 GT impression과 clinically aligned 되도록 critic feedback을 만든다.

critic:

- Qwen3.5-9B 같은 critic model이 prediction vs GT를 비교해 natural language feedback을 준다.

acceptance gate:

- clinical F1이 baseline보다 유의미하게 떨어지면 reject
- lesion agreement vs baseline이 너무 낮으면 reject
- normal collapse rate가 증가하면 reject
- hallucination case가 증가하면 reject

현재 코드 위치:

- placeholder runner: `scripts/05_run_textgrad_prompt_optimization.py`
- optimization wrapper: `src/vision_rag_cxr/prompting/textgrad_optimizer.py`
- experiment placeholder: `src/vision_rag_cxr/experiments/prompt_optimization_experiment.py`
- prompt templates: `src/vision_rag_cxr/prompting/prompt_templates.py`

## 10. Vision encoder 선택 실험

후보:

- MedSigLIP: image-text common embedding retrieval 후보
- MedCLIP: medical image-text contrastive retrieval baseline
- BioViL: radiology-specific image-text representation baseline
- CheXzero: CXR report/image contrastive zero-shot baseline

선택 metric:

- support/inference retrieval sanity: same-label retrieval rate, label Jaccard@k
- image-only nearest neighbor clinical relevance
- downstream RAG improvement: related RAG impression/localization gain
- runtime/memory cost

권장 DB granularity:

- 14-label만 쓰면 coarse하다.
- final DB에는 image embedding + impression text embedding + label vector + anatomy/pathology phrase를 모두 저장한다.
- 검색은 image embedding 중심, ranking/debug은 label/anatomy phrase를 보조로 둔다.

현재 코드 위치:

- encoder interface: `src/vision_rag_cxr/models/base.py`
- encoder factory: `src/vision_rag_cxr/models/vision_encoder_base.py`
- adapters: `src/vision_rag_cxr/models/medsiglip_encoder.py`, `medclip_encoder.py`, `biovil_encoder.py`, `chexzero_encoder.py`

## 11. Final 실험

Final paired comparison:

1. No-RAG: `Image + optimized prompt -> impression`
2. Vision-RAG: `Image -> selected vision encoder -> top-k support examples -> Image + optimized prompt + k-shot image/impression examples -> impression`

같은 inference uid에 대해 두 condition을 모두 생성하고 paired metric으로 비교한다.

현재 코드 위치:

- final runner: `scripts/07_run_final_rag_experiment.py`
- final class: `src/vision_rag_cxr/experiments/final_rag_experiment.py`
- config: `configs/experiments/final_rag_vs_no_rag.yaml`

## 12. Plug-and-play 수정 위치

| 바꾸고 싶은 것 | 수정 파일 |
|---|---|
| 데이터 경로/projection metadata | `configs/data/indiana.yaml` |
| label source/uncertain policy | `configs/data/indiana.yaml`, `src/vision_rag_cxr/data/labeler_chexbert.py` |
| support/inference 비율/seed | `configs/split/stratified_70_30.yaml` |
| generator model | `configs/models/medgemma.yaml`, `src/vision_rag_cxr/models/vlm_generator.py` |
| MedGemma 실제 HF generation | `src/vision_rag_cxr/models/medgemma_generator.py` |
| critic model | `configs/models/qwen35_9b_critic.yaml`, `src/vision_rag_cxr/models/qwen_critic.py` |
| retrieval encoder | `configs/retrieval/hybrid_rag.yaml`, `src/vision_rag_cxr/models/*_encoder.py` |
| related/unrelated retrieval policy | `src/vision_rag_cxr/rag/related_retriever.py`, `unrelated_retriever.py` |
| prompt schema | `src/vision_rag_cxr/prompting/prompt_templates.py` |
| TextGrad objective | `src/vision_rag_cxr/prompting/textgrad_optimizer.py` |
| localization metric | `src/vision_rag_cxr/evaluation/localization_metrics.py` |
| report metric | `src/vision_rag_cxr/evaluation/chexbert_metrics.py`, `report_metrics.py` |

## 13. 현재 구현 상태 audit

작동 확인됨:

- 전처리 script가 실제 `/root/data`에서 3,405 paired samples를 생성함.
- split script가 7:3 multilabel stratified split을 생성함.
- RAG DB builder가 dummy_hash embedding으로 support DB를 생성함.
- RelatedRetriever가 실제 image embedding top-k를 반환함.
- RAG ablation experiment가 retrieved_uids/retrieval_scores를 predictions.csv에 저장함.

주의/미완성:

- MedGemma generator는 아직 framework placeholder이다. production 결과에는 HF generation 연결이 필요하다.
- TextGrad experiment는 placeholder이다. 실제 textgrad objective와 critic loop 연결이 필요하다.
- MedSigLIP/MedCLIP/BioViL/CheXzero adapter는 interface skeleton이다. 실제 model loading/encoding 구현이 필요하다.
- IU-only bbox metric은 human audit 기반으로 보고해야 한다.

## 14. 바로 사용할 수 있는 코드 생성 프롬프트

다음 프롬프트를 새 Codex/LLM 세션에 넣으면 된다.

```text
너는 의료 Vision-Language RAG 연구용 Python repository를 구현하는 senior ML research engineer다.

목표는 Indiana University Chest X-ray dataset 기반 Vision-RAG CXR 실험 프레임워크를 E2E로 완성하는 것이다. 현재 저장소는 /root/vision-rag-cxr이며, 데이터는 /root/data/images/images_normalized, 라벨 CSV는 /root/data/label/indiana_reports.csv에 있다. 프레임워크는 config-driven, modular, plug-and-play 구조여야 하며, raw data는 절대 덮어쓰면 안 된다.

반드시 유지할 연구 정책:
1. uid 단위로 frontal+lateral image pair가 모두 있는 sample만 사용한다.
2. label space는 CheXpert/CheXbert 14 labels이다.
3. primary label source는 impression, secondary는 findings+impression, MeSH/Problems는 auxiliary audit source이다.
4. main label은 multi-label로 유지하고 top-1로 축소하지 않는다.
5. support:inference split은 7:3이고 multilabel iterative stratification으로 label drift를 낮춘다.
6. final/no-leakage retrieval은 query image embedding만 사용한다. GT CheXbert label을 retrieval query에 쓰는 설정은 oracle/debug으로만 허용한다.
7. IU-only bbox 결과는 strict mAP가 아니라 human audit localization correctness로 보고한다.
8. TextGrad는 MedGemma weight를 수정하지 않고 STYLE_PROFILE prompt variable만 optimize한다.
9. final 실험은 같은 inference uid에 대해 No-RAG와 Vision-RAG를 paired comparison한다.

필수 구현/점검 순서:
1. scripts/01_preprocess_indiana.py가 configs/data/indiana.yaml을 읽고 outputs/preprocessed에 canonical CSV/JSONL/report를 저장하는지 확인한다.
2. scripts/02_create_stratified_split.py가 seed별 split과 split quality report를 저장하는지 확인한다.
3. scripts/03_build_support_database.py가 support DB metadata, image/text embeddings, label vectors를 저장하는지 확인한다.
4. src/vision_rag_cxr/rag/related_retriever.py가 실제 top-k related examples를 반환하고, unrelated_retriever.py가 negative control examples를 반환하는지 확인한다.
5. scripts/04_run_baseline_experiments.py와 scripts/06_run_rag_ablation.py가 retrieved_uids/retrieval_scores를 predictions.csv에 저장하는지 확인한다.
6. MedGemma adapter에 실제 Hugging Face generation을 연결하되, interface는 BaseGenerator를 유지한다.
7. TextGrad wrapper는 원 논문 구조를 크게 건드리지 말고 STYLE_PROFILE variable, Qwen critic, acceptance gate만 chest xray domain adapter로 분리한다.
8. evaluation 모듈에서 CheXbert clinical metrics, natural language metrics, paired t-test/Wilcoxon/bootstrap CI를 생성한다.
9. Docker GPU 환경에서 동일 command로 실행 가능하게 유지한다.
10. README와 docs/PROJECT_FRAMEWORK_PROMPT_KO.md를 최신 실행 흐름과 plug-and-play 수정 위치에 맞춰 갱신한다.

실행 검증 command:
python -m py_compile $(rg --files src scripts | rg '\.py$')
python scripts/01_preprocess_indiana.py --config configs/data/indiana.yaml
python scripts/02_create_stratified_split.py --data_csv outputs/preprocessed/indiana_paired_samples.csv --config configs/split/stratified_70_30.yaml
python scripts/03_build_support_database.py --split_csv outputs/splits/split_seed_0.csv --config configs/retrieval/hybrid_rag.yaml
python scripts/04_run_baseline_experiments.py --config configs/experiments/exp_01_image_only_localization.yaml
python scripts/06_run_rag_ablation.py --config configs/experiments/exp_05_related_rag_localization.yaml
python scripts/07_run_final_rag_experiment.py --config configs/experiments/final_rag_vs_no_rag.yaml
```
