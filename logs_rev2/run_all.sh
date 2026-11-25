#!/bin/bash
cd /Users/murad/Documents/Thesis_work/DL_Application_for_Landslide_Susceptibility_Mapping
PY=/Users/murad/miniforge3/envs/geo-ml/bin/python
date +%s > logs_rev2/start_time
for batch in "42 0 1 2 3" "4 5 6 7 8"; do
  for s in $batch; do
    $PY -m src.revision_analyses --part seeds --seeds $s > logs_rev2/seed$s.log 2>&1 &
  done
  wait
  date +%s > logs_rev2/batch_done_$(echo $batch | cut -d' ' -f1)
done
date +%s > logs_rev2/end_time
