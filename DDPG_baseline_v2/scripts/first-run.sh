#!/usr/bin/env bash

NAMES=(a b c d e f g h i j k l m n o p q r s t)

for NAME in ${NAMES[*]}
do
  export LOGS=logs/first/$NAME
  rm -rf $LOGS
  mkdir -p $LOGS
  (
    export LOGS
    export NAME
    export PERF_STUDY="xfirst_$NAME"
    rm -f $LOGS/${PERF_STUDY}.e* $LOGS/${PERF_STUDY}.o* ${PERF_STUDY}.e* ${PERF_STUDY}.o*
    qsub -N ${PERF_STUDY} -o "$LOGS/${PERF_STUDY}.out" -b "$LOGS/${PERF_STUDY}.err" -d . scripts/first-submit.sh
  )
done
