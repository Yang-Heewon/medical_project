#!/usr/bin/env bash
# 3-GPU 병렬 sweep: GPU 0,2,3 (GPU 1 결함 제외). split/DB는 한 번만 prebuild 후 3 shard 병렬.
set -u
cd /root/research/heewon/medical_project
source /root/anaconda3/etc/profile.d/conda.sh && conda activate vision_rag_cxr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/sweep/real_iu_sweep.yaml
echo "[mg] $(date) prebuild split+DB (GPU0) ..."
CUDA_VISIBLE_DEVICES=0 python -u scripts/16_run_experiment_sweep.py --config "$CFG" --build_only
GPUS=(0 2 3)
for i in 0 1 2; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} python -u scripts/16_run_experiment_sweep.py --config "$CFG" --shard_id "$i" --num_shards 3 > logs/sweep_shard_$i.log 2>&1 &
  echo "[mg] shard $i -> GPU ${GPUS[$i]} PID=$!"
done
wait
echo "[mg] $(date) all shards complete"
