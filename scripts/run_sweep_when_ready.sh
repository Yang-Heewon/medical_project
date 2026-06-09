#!/usr/bin/env bash
# 데이터 빌드 완료를 기다렸다가 sweep을 실행하는 무인 런처.
set -u
cd /root/research/heewon/medical_project
source /root/anaconda3/etc/profile.d/conda.sh && conda activate vision_rag_cxr
export CUDA_VISIBLE_DEVICES=0   # GPU 1(결함) 회피, 단일 V100
echo "[launcher] waiting for /root/data/real_iu/BUILD_DONE ..."
while [ ! -f /root/data/real_iu/BUILD_DONE ]; do sleep 60; done
echo "[launcher] data ready: $(cat /root/data/real_iu/BUILD_DONE) samples. starting sweep."
python -u scripts/16_run_experiment_sweep.py --config configs/sweep/real_iu_sweep.yaml
echo "[launcher] sweep process exited."
