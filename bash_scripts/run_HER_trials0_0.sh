#!/bin/sh
#SBATCH --mincpus 20
#SBATCH -p court
#SBATCH -t 3:00:00
#SBATCH -e run_HER_trials0_0.sh.err
#SBATCH -o run_HER_trials0_0.sh.out
rm log.txt; 
export EXP_INTERP='/home/ccolas/virtual_envs/py3.5/bin/python' ;
echo '=================> Her : Trial 0, 2018-06-22 15:11:35.444320';
echo '=================> Her : Trial 0, 2018-06-22 15:11:35.444320' >> log.txt;
$EXP_INTERP train.py --active_goal False --replay_strategy future --logdir ./save/ArmBall-v0/0/ 
wait