#!/bin/bash
PY=/Users/murad/miniforge3/envs/geo-ml/bin/python
$PY -m src.revision_rev2 --part extra_models --seeds $1 > logs_rev2b/catboost_seed$1.log 2>&1
