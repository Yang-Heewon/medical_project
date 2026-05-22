# 다른 서버에서 실행하기

이 문서는 GitHub에서 새 서버로 clone한 뒤 Vision-RAG-CXR smoke test와 prompt-preservation pilot을 재현하는 절차입니다.

## 1. Clone

```bash
git clone https://github.com/Yang-Heewon/medical_project.git
cd medical_project
```

## 2. 데이터 배치

기본 config는 아래 경로를 사용합니다.

```text
/root/data/images/images_normalized
/root/data/label/indiana_reports.csv
```

다른 위치를 쓴다면 `configs/data/indiana.yaml`의 경로를 바꾸세요.

필수 데이터:

- Indiana CXR image PNG files
- Indiana reports CSV: `uid,MeSH,Problems,image,indication,comparison,findings,impression`

## 3. Python 환경

권장: CUDA가 맞는 conda/venv를 먼저 만든 뒤 설치합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

서버에 이미 `/root/miniconda3/envs/llm_rag` 같은 GPU 환경이 있으면 그 Python을 써도 됩니다.

## 4. 모델 준비

실제 MedGemma/BiomedCLIP smoke를 돌리려면 Hugging Face cache 또는 인터넷 접근이 필요합니다.

현재 config:

- MedGemma: `configs/models/medgemma_real.yaml`
- BiomedCLIP: `configs/retrieval/hybrid_rag_biomedclip.yaml`

MedGemma 접근 권한이 필요한 경우 Hugging Face login을 먼저 수행하세요.

```bash
huggingface-cli login
```

## 5. CheXbert 준비

production label을 만들려면 외부 CheXbert repo/checkpoint가 필요합니다.

기본 기대 경로:

```text
/root/CheXbert/run_chexbert.py
/root/CheXbert/chexbert.pth
```

이미 `outputs/production/preprocessed/indiana_paired_samples_chexbert.csv`가 있으면 one-click script는 CheXbert 생성을 skip합니다. GitHub에는 outputs가 올라가지 않으므로 새 서버에서는 CheXbert를 준비하거나, 해당 CSV를 별도로 복사해야 합니다.

## 6. 가장 빠른 pipeline smoke

```bash
python -u scripts/00_run_e2e_pipeline.py   --config configs/pipeline/e2e_smoke.yaml   2>&1 | tee logs/e2e_smoke.log
```

이 smoke는 구조 검증용이며 기본 dummy/placeholder 경로를 포함할 수 있습니다.

## 7. One-click 검증

아래 명령은 빠른 E2E smoke, CheXbert label 준비 확인, production split, MedGemma prompt-preservation sampling t-test를 한 번에 실행합니다.

```bash
RUN_NAME=first_check GPU_IDS=1,2,3,4,6 SAMPLE_SIZE=8 bash scripts/run_one_click_smoke_and_prompt_check.sh
```

실시간 로그:

```bash
tail -f logs/first_check/prompt_preservation_sampling_ttest.log
```

결과:

```text
outputs/first_check/prompt_preservation_sample/sample_metric_summary.json
outputs/first_check/prompt_preservation_sample/sample_lesion_scores.csv
outputs/first_check/prompt_preservation_sample/sample_predictions.csv
outputs/first_check/prompt_preservation_sample/sample_audit.html
```

## 8. 실제 모델 연결 smoke 옵션

현재 `RUN_REAL_SMOKE=1`은 MedGemma/BiomedCLIP wiring 확인용입니다. 아직 최종 MeSH/Problems controlled k-shot design으로 완전히 교체되기 전입니다.

```bash
RUN_NAME=real_wiring_check GPU_IDS=1,2,3,4,6 SAMPLE_SIZE=3 RUN_REAL_SMOKE=1 bash scripts/run_one_click_smoke_and_prompt_check.sh
```

## 9. GitHub에 올라가지 않는 것

`.gitignore`에 의해 다음은 push되지 않습니다.

- `outputs/`
- `logs/`
- `.env`
- model checkpoints: `*.pth`, `*.safetensors`, `*.bin`
- local data: `data/`, `checkpoints/`

따라서 새 서버에서는 데이터/모델/outputs를 별도로 준비해야 합니다.

## 10. 현재 중요한 연구 상태

- CheXbert 14-label 기반 split/evaluation 경로가 있습니다.
- MedGemma real adapter가 있습니다.
- BiomedCLIP/OpenCLIP adapter가 있습니다.
- prompt-preservation sampling t-test와 HTML audit 생성 스크립트가 있습니다.
- 다음 큰 수정은 3-6번 ablation을 MeSH/Problems controlled k-shot DB로 교체하는 것입니다.
