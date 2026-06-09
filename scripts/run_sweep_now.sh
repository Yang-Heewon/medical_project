#!/usr/bin/env bash
set -u
cd /root/research/heewon/medical_project
source /root/anaconda3/etc/profile.d/conda.sh && conda activate vision_rag_cxr
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/16_run_experiment_sweep.py --config configs/sweep/real_iu_sweep.yaml
echo "[sweep] process exited."
