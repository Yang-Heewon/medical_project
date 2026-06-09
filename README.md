# vision-rag-cxr

Chest X-ray **Vision-RAG** 실험 프레임워크. 핵심 질문은 하나입니다:

> **다양한 데이터셋 × 다양한 VLM에서, related example을 retrieval해 넣는 Vision-RAG가 보고서 생성을 실제로 개선하는가?** (No-RAG vs Vision-RAG, CheXbert F1로 측정)

데이터셋·생성모델·인코더가 모두 **plug-in/plug-out** 레지스트리로 되어 있어 이름만 바꿔 끼웁니다.

---

## 3가지 옵션 (통합 CLI `vrag`)

```bash
pip install -e .          # vrag 콘솔 명령 설치 (또는 python scripts/vrag.py ...)
```

### ⓪ 무엇을 끼울 수 있나 — `vrag list`
```bash
vrag list
```
dataset / generator / encoder plug-in 카탈로그와 각 제약(주의)을 출력합니다.

### ① 데이터 구축 — `vrag build`
canonical schema(아래)로 데이터셋을 만듭니다.
```bash
# 무인증으로 실제 Indiana IU 전체를 HF 스트리밍으로 구축
vrag build --dataset indiana_hf --out /root/data/real_iu

# 로컬 Indiana CSV+이미지에서
vrag build --dataset indiana --config configs/data/indiana.yaml --out outputs/iu

# PadChest-GR (BIMCV 승인 데이터 필요)
vrag build --dataset padchest_gr --config <padchest.yaml> --out outputs/pcgr
```

### ② inference (No-RAG vs Vision-RAG) — `vrag infer`
```bash
vrag infer \
  --dataset-csv /root/data/real_iu/real_iu_paired.csv \
  --generators qwen2.5-vl \
  --encoder biomedclip \
  --modes no_rag,related \
  --gpus 0,2,3 \
  --out outputs/infer_iu
```
- `--generators`는 쉼표로 여러 개(매트릭스). `--modes`는 `no_rag,related,unrelated`.
- 결과: `outputs/.../results*.csv` 에 조합별 **CheXbert micro/macro F1** 누적. resumable + 조합 실패 격리.
- `--gpus 0,2,3` 처럼 여러 GPU를 주면 조합을 샤딩해 병렬 실행합니다.

---

## Plug-in 카탈로그 (현재)

| 종류 | 사용 가능 | 비고 |
|---|---|---|
| **dataset** | `indiana`, `indiana_hf`(무인증) | `padchest_gr` (24-label+bbox GT, **BIMCV 승인 필요**) |
| **generator** | `qwen2.5-vl`, `medgemma`(gated), `placeholder` | `llava-med` (model_type `llava_mistral` — 표준 transformers 로드 불가, 별도 LLaVA 코드 필요) |
| **encoder** | `biomedclip`(open_clip), `dummy_hash` | `medsiglip`/`biovil`/`chexzero`는 skeleton |

새 어댑터 추가 위치: `src/vision_rag_cxr/{datasets, models/generators, models/encoders}/` + `registries.py` 카탈로그 한 줄.

---

## 패키지 구조

```
src/vision_rag_cxr/
  cli.py            # vrag {list|build|infer} 진입점
  registries.py     # 중앙 plug 카탈로그 + build_* 팩토리 재노출
  datasets/         # ① 데이터 구축: registry, indiana, indiana_hf, padchest_gr,
                    #    labeler_chexbert, label_spaces, splitters
  models/           # ③ plug-in 모델: generators/, encoders/, critics/, base
  inference/        # ② inference: sweep, experiments/, retrieval/
  evaluation/  prompting/  utils/
```

## Canonical schema (모든 dataset adapter 공통 출력)
`uid, frontal_path, lateral_path, impression, findings, MeSH, Problems, chexbert_labels_binary(JSON), chexbert_labels_raw(JSON), anatomy_pathology_phrase` (+ bbox GT가 있으면 `localization_gt`).
이 스키마만 맞추면 split·RAG DB·실험·평가가 그대로 동작합니다.

## Label space
`label_space` 레지스트리로 데이터셋별 라벨 체계를 고릅니다: `chexbert_14`(IU 기본), `padchest_gr_24`(PadChest-GR 논문 24 finding). 산출물에 `label_space.json` sidecar로 기록되어 retriever/metric이 자동 복원합니다.

## 환경 노트
- V100(sm_70): `torch==2.6.0+cu124`(fp16), `transformers<5` 고정. 최신 cu128/cu130 휠과 transformers 5.x는 각각 Volta/BiomedCLIP·Qwen 로딩과 비호환.

## 레거시 단계별 스크립트
`scripts/00_run_e2e_pipeline.py` 외 단계별 스크립트(01~16)도 그대로 동작합니다. 일상 사용은 `vrag` CLI를 권장합니다.
