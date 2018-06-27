#!/usr/bin/env bash

#PBS -l walltime=00:59:59
# Request a node per experiment to avoid competition between different TFs
#PBS -l nodes=1:ppn=24
#PBS -V
NB_TRIALS=50

for TRIAL in $(seq $NB_TRIALS)
do
  (
    echo "Running experiment $TRIAL"
    export TRIAL
    python baselines/ddpg/main_first.py > ${LOGS}/${PERF_STUDY}_${TRIAL}.out 2> ${LOGS}/${PERF_STUDY}_${TRIAL}.err
  ) &
done

wait
