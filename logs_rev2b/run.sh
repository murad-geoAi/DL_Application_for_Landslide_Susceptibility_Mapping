#!/bin/bash
PY=/Users/murad/miniforge3/envs/geo-ml/bin/python
s=$1
export OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
for p in spatial_es lstm_reg ablation_more; do
  /usr/bin/time -p $PY -m src.revision_rev2 --part $p --seeds $s > logs_rev2b/${p}_seed$s.log 2>&1
done
