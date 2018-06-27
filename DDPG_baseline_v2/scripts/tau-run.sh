#!/usr/bin/env bash

TAUS=(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0 5.0 10.0 50.0 100.0)

for TAU in ${TAUS[*]}
do
  export LOGS=logs/tau/$TAU
  rm -rf $LOGS
  mkdir -p $LOGS
  (
    export LOGS
    export TAU
    export PERF_STUDY="xtau_$TAU"
    rm -f $LOGS/${PERF_STUDY}.e* $LOGS/${PERF_STUDY}.o* ${PERF_STUDY}.e* ${PERF_STUDY}.o*
    qsub -N ${PERF_STUDY} -o "$LOGS/${PERF_STUDY}.out" -b "$LOGS/${PERF_STUDY}.err" -d . scripts/perf-submit_tau.sh
  )
done
