#!/usr/bin/env bash
# textgrad-first 순서: TextGrad 끝나면 최적화 프롬프트로 No-RAG/Vision-RAG infer.
set -u
cd /root/research/heewon/medical_project
source /root/anaconda3/etc/profile.d/conda.sh && conda activate vision_rag_cxr
SP=outputs/textgrad_qwen/optimized_style_profile.txt
echo "[chain] $(date) TextGrad 완료 대기..."
while ps aux | grep -E "05_run_textgrad" | grep -v grep >/dev/null; do sleep 30; done
sleep 5
if [ ! -f "$SP" ]; then echo "[chain] ERROR: $SP 없음 (TextGrad 실패?)"; exit 1; fi
echo "[chain] TextGrad 완료. 최적화 프롬프트:"; cat "$SP"
echo "[chain] $(date) 최적화 프롬프트로 infer 실행 (GPU 0,2,3)"
rm -rf outputs/infer_iu_opt
vrag infer --dataset-csv /root/data/real_iu/real_iu_paired.csv \
  --generators qwen2.5-vl --encoder biomedclip --modes no_rag,related \
  --device auto --gpus 0,2,3 --max-samples 0 \
  --style-profile "$SP" --out outputs/infer_iu_opt
echo "[chain] $(date) infer 완료"
