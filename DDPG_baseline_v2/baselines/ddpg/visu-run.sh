#!/usr/bin/env bash

VISUS=(0.01 0.05 0.1 0.5 1.0 5.0 10.0)

for VISU in ${VISUS[*]}
do
  export LOGS=logs/visu/$VISU
  rm -rf $LOGS
  mkdir -p $LOGS
  (
    export LOGS
    export TAU
    export PERF_STUDY="xvisu_$VISU"
    rm -f $LOGS/${PERF_STUDY}.e* $LOGS/${PERF_STUDY}.o* ${PERF_STUDY}.e* ${PERF_STUDY}.o*
    qsub -N ${PERF_STUDY} -o "$LOGS/${PERF_STUDY}.out" -b "$LOGS/${PERF_STUDY}.err" -d . visu-submit.sh
  )
done
