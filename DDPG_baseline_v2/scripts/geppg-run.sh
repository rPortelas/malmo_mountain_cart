#!/usr/bin/env bash

NAMES=(a b c d e f g)

for NAME in ${NAMES[*]}
do
  export LOGS=logs/geppg/$NAME
  rm -rf $LOGS
  mkdir -p $LOGS
  (
    export LOGS
    export NAME
    export PERF_STUDY="xgeppg_$NAME"
    rm -f $LOGS/${PERF_STUDY}.e* $LOGS/${PERF_STUDY}.o* ${PERF_STUDY}.e* ${PERF_STUDY}.o*
    qsub -N ${PERF_STUDY} -o "$LOGS/${PERF_STUDY}.out" -b "$LOGS/${PERF_STUDY}.err" -d . scripts/geppg-submit.sh
  )
done
