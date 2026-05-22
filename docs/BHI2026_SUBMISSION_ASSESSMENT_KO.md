# BHI 2026 제출 적합성 분석

## 결론

이 프로젝트는 IEEE EMBS BHI 2026에 제출할 만한 주제 적합성이 높습니다. 특히 다음 BHI 2026 topic과 잘 맞습니다.

- Generative, Cognitive, Explainable and Causal AI for Health Informatics
- Multimodal Biomedical Data Fusion
- AI Implementation and Clinical Translation
- Precision and Predictive Healthcare

핵심 포지셔닝은 `Chest X-ray VLM의 prompt optimization과 controlled Vision-RAG가 병변 보존/보고서 생성에 미치는 영향`입니다.

## 제출 가능성이 높은 논문 각도

권장 제목 방향:

> Lesion-Preserving Prompt Optimization and Controlled Vision-RAG for Chest X-ray Report Generation

또는

> Evaluating Vision-RAG for Chest X-ray VLMs with Lesion-Preservation Constraints

## BHI에 맞는 이유

BHI는 의료 informatics, multimodal biomedical data, generative/explainable AI, clinical translation을 다룹니다. 이 프로젝트는 CXR image, radiology text, MeSH/Problems, CheXbert labels를 결합하고, VLM output의 병변 보존성을 통계적으로 검증한다는 점에서 BHI scope와 잘 맞습니다.

## 반드시 강화해야 할 점

현재 상태 그대로는 workshop/demo/pilot으로는 가능하지만, regular paper로 강하게 내려면 다음이 필요합니다.

1. MeSH/Problems controlled support DB
   - 3~6번 ablation에서 related/unrelated k-shot을 image-nearest가 아니라 MeSH/Problems 병변-위치 기준으로 통제해야 합니다.

2. Prompt optimization의 lesion-preservation gate
   - baseline prompt vs optimized prompt를 같은 uid에서 비교합니다.
   - paired t-test만 쓰지 말고 non-inferiority margin을 같이 제시합니다.

3. 14-label multi-label evaluation
   - CheXbert label 기준 micro/macro F1, precision, recall, hamming loss, subset accuracy를 제시합니다.

4. Human-loop localization audit
   - Indiana에는 bbox GT가 없으므로 bbox IoU를 정답처럼 주장하면 안 됩니다.
   - HTML overlay audit 결과를 subset에 대해 보고해야 합니다.

5. Normal collapse / abnormal miss 분석
   - VLM이 normal로 과도하게 수렴하는지 보여주는 분석이 중요합니다.

## 권장 실험 구성

### Experiment A: Baseline VLM lesion localization

Image + base localization prompt -> bbox/lesion JSON

목적:
- VLM 자체 병변 탐지 능력 측정
- prompt optimization 전 baseline 저장

### Experiment B: Prompt optimization with lesion preservation

Image + base prompt vs Image + optimized prompt

평가:
- impression similarity / CheXbert agreement 개선
- lesion score non-inferiority 통과 여부

### Experiment C: Controlled k-shot ablation

Prompt는 고정하고 k-shot만 바꿉니다.

1. Image only -> bbox
2. Image only -> impression
3. Image + unrelated MeSH/Problems image+impression examples -> bbox
4. Image + unrelated MeSH/Problems image+impression examples -> impression
5. Image + related MeSH/Problems image+impression examples -> bbox
6. Image + related MeSH/Problems image+impression examples -> impression

### Experiment D: Final Vision-RAG retrieval

Vision encoder가 image만 보고 MeSH/Problems-like related bucket을 얼마나 잘 찾아오는지 평가합니다.

## 주의: Double-blind

BHI regular paper는 double-blind입니다. 논문 본문에는 identifying GitHub link를 넣으면 안 됩니다. 공개 GitHub repo는 camera-ready 또는 accepted 이후 artifact로 연결하는 편이 안전합니다. 제출 시에는 anonymous supplementary 또는 anonymized artifact를 사용하세요.

## 현재 readiness

- 1-page abstract: 비교적 빠르게 가능
- regular paper: MeSH/Problems controlled k-shot DB와 human audit 결과를 추가한 뒤 권장

## 추천 전략

1. 우선 regular paper 목표로 실험을 강화합니다.
2. 6월 초까지 controlled k-shot 결과와 prompt-preservation gate 결과를 정리합니다.
3. regular paper가 부족하면 1-page abstract fallback을 준비합니다.
