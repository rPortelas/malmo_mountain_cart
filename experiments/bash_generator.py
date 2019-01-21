#!/usr/bin/python
# -*- coding: utf-8 -*-
PATH_TO_INTERPRETER = "/cm/shared/apps/intel/composer_xe/python3.5/intelpython3/bin/python3"  # plafrim

save_dir = '../'
env = 'Maze-v0'
trial_id = list(range(0, 20))
algo = 'random_modular'
experiment_name = 'armtrajplaf'
nb_iter = '100000'
nb_bt = '5'
distractors = 'False'
trajectories = 'False'
explo_noise = 0.005
#update_interest_step = 5

filename = save_dir + 'run_' + algo + '_' + experiment_name + str(trial_id[0]) + '_' + str(trial_id[-1]) + '.sh'

with open(filename, 'w') as f:
    f.write('#!/bin/sh\n')
    f.write('#SBATCH --mincpus 20\n')
    f.write('#SBATCH -p court \n')
    f.write('#SBATCH -t 4:00:00\n')
    f.write('#SBATCH -e ' + filename[1:] + '.err\n')
    f.write('#SBATCH -o ' + filename[1:] + '.out\n')
    f.write('rm log.txt; \n')
    f.write("export EXP_INTERP='%s' ;\n" % PATH_TO_INTERPRETER)

    for seed in range(len(trial_id)):
        t_id = trial_id[seed]
        f.write("$EXP_INTERP imgep_in_armtoolstoys.py %s_%s %s %s %s %s %s %s &\n" % (experiment_name,t_id, algo, nb_iter, nb_bt, explo_noise, distractors, trajectories))
    f.write('wait')
