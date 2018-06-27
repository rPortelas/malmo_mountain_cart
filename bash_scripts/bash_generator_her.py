
#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime

PATH_TO_INTERPRETER = "/home/ccolas/virtual_envs/py3.5/bin/python"

save_dir = './save/'
env = 'ArmBall-v0'
trial_id = list(range(0, 1))
study = 'HER'  #'DDPG'  #'DDPG'
active_goal = False
replay_strategy = 'future'

filename = 'run_' + study + '_trials' + str(trial_id[0])+'_'+str(trial_id[-1])+'.sh'
with open(filename, 'w') as f:
    f.write('#!/bin/sh\n')
    f.write('#SBATCH --mincpus 20\n')
    f.write('#SBATCH -p court\n')
    f.write('#SBATCH -t 3:00:00\n')
    f.write('#SBATCH -e ' + filename + '.err\n')
    f.write('#SBATCH -o ' + filename + '.out\n')
    f.write('rm log.txt; \n')
    f.write("export EXP_INTERP='%s' ;\n" % PATH_TO_INTERPRETER)

    for seed in range(len(trial_id)):
        t_id = trial_id[seed]
        name = (study+" : trial %s, %s" % (str(t_id), str(datetime.datetime.now()))).title()
        f.write("echo '=================> %s';\n" % name)
        f.write("echo '=================> %s' >> log.txt;\n" % name)
        f.write("$EXP_INTERP train.py --active_goal %s --replay_strategy %s --logdir %s \n" % (str(active_goal), replay_strategy, save_dir + env + '/'+ str(t_id)+'/'))

    f.write('wait')
