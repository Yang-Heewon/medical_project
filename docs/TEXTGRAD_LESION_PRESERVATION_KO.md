# TextGrad Lesion Preservation Gate

목표는 TextGrad/Qwen critic으로 impression prompt를 개선하되, MedGemma 원래 병변 탐지 능력을 통계적으로 해치지 않는 prompt만 채택하는 것이다.

## 왜 단순 p > 0.05가 부족한가

paired t-test에서 p > 0.05는 "차이가 없음을 증명"하는 것이 아니다. 표본이 작거나 variance가 크면 실제로 나빠져도 유의하지 않게 나올 수 있다.

따라서 이 프로젝트에서는 margin을 정한 non-inferiority 또는 equivalence test를 쓴다.

권장 기본값:

```yaml
lesion_preservation_gate:
  enabled: true
  mode: noninferiority
  margin: 0.03
  alpha: 0.05
```

해석:

- baseline = MedGemma 기본 prompt의 병변 localization score
- candidate = TextGrad optimized prompt의 병변 localization score
- delta = candidate - baseline
- non-inferiority H0: delta <= -0.03
- 통과 조건: candidate가 baseline보다 평균 3 percentage point 이상 나빠졌다고 보기 어렵고, one-sided paired t-test를 통과해야 한다.

## production loop

1. baseline MedGemma prompt로 dev set localization 실행
2. TextGrad/Qwen critic이 candidate STYLE_PROFILE 생성
3. candidate STYLE_PROFILE로 같은 dev set impression 생성
4. candidate STYLE_PROFILE로 같은 dev set localization 실행
5. 같은 uid 순서로 baseline/candidate lesion score CSV 생성
6. `evaluate_lesion_preservation_ttest()` 실행
7. clinical metric과 lesion gate가 모두 통과하면 prompt 채택

## score CSV 형식

```csv
uid,baseline_lesion_score,candidate_lesion_score
1,1.0,1.0
2,0.5,0.5
3,0.0,0.0
```

score는 나중에 사용할 metric에 따라 바꿀 수 있다.

후보:

- human audit bbox correctness: 0/1
- pseudo/manual bbox IoU correctness: 0/1
- lesion label F1 per case: 0~1
- label + anatomy + bbox combined score: 0~1

## 코드 위치

- gate function: `src/vision_rag_cxr/prompting/textgrad_optimizer.py`
- prompt optimization runner: `src/vision_rag_cxr/experiments/prompt_optimization_experiment.py`
- config: `configs/experiments/textgrad_prompt_optimization.yaml`

## 중요한 설계

bbox prompt를 TextGrad가 직접 optimize하는 것이 아니다. localization prompt는 고정된 검사 도구처럼 사용한다. TextGrad는 impression용 STYLE_PROFILE을 바꾸고, 그 결과가 lesion localization 성능을 망치지 않는지 같은 dev uid에서 검정한다.
