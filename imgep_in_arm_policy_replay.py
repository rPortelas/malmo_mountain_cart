from builtins import range
import os
import os.path
import sys
import time
import json
import pickle
import time
import numpy as np
from gep import GEP
from utils.neural_network import PolicyNN
import matplotlib.pyplot as plt
from utils.plot_utils import *
import collections
from collections import OrderedDict
from utils.gep_utils import *
#import gym2
import gym
import gym_flowers
import config
#import cProfile


def get_outcome(state, distractors):
    return get_state(state, distractors)

def get_state(state, distractors):
    if distractors:
        s = state
    else:
        s = state[0:9] + state[13:15]
    return np.array(s)

def run_episode(model, distractors):
    out = arm_env.reset()
    state = out['observation']
    # Loop until mission/episode ends:
    done = False
    while not done:
        # extract the world state that will be given to the agent's policy
        normalized_state = scale_vector(get_state(state, distractors), np.array(input_bounds))
        actions = model.get_action(normalized_state.reshape(1, -1))
        out, _, done, _ = arm_env.step(actions[0])
        #arm_env.render()
        state = out['observation']
    return get_outcome(state, distractors)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


#cp = cProfile.Profile()
#cp.enable()

# define and parse argument values
# more info here: https://stackoverflow.com/questions/5423381/checking-if-sys-argvx-is-defined
distractors = False
if not distractors:
    state_names = ['hand_x', 'hand_y', 'gripper', 'stick1_x', 'stick1_y', 'stick2_x', 'stick2_y',
     'magnet1_x', 'magnet1_y', 'scratch1_x', 'scratch1_y']
else:
    state_names = ['hand_x', 'hand_y', 'gripper', 'stick1_x', 'stick1_y', 'stick2_x', 'stick2_y',
                   'magnet1_x', 'magnet1_y', 'magnet2_x', 'magnet2_y', 'magnet3_x', 'magnet3_y',
                   'scratch1_x', 'scratch1_y', 'scratch2_x', 'scratch2_y', 'scratch3_x', 'scratch3_y',
                   'cat_x', 'cat_y', 'dog_x','dog_y',
                   'static1_x','static1_y', 'static2_x','static2_y', 'static3_x','static3_y', 'static4_x','static4_y']
b = config.get_env_bounds('arm_env')

# init neural network policy
input_names = state_names
input_bounds = b.get_bounds(input_names)
input_size = len(input_bounds)
print('input_bounds: %s' % input_bounds)
layers = [64]
action_set_size = 4
params = {'layers': layers, 'activation_function':'relu', 'max_a':1.,
          'dims':{'s':input_size,'a':action_set_size},'bias':True}
param_policy = PolicyNN(params)
total_policy_params = get_n_params(param_policy)
print('nbparams:    {}'.format(total_policy_params))

#####################################################################
# init arm_env controller
arm_env = env=gym.make('ArmToolsToys-v0')

with open("tmpuuuu_policy_stick1_1547753244.1490777.pickle", 'rb') as handle:
    policy_params = pickle.load(handle)
outs=[]
add_noise = True
noise = 0.005
for i in range(0, 100):
    #print(policy_params[0])
    if add_noise:
        noised_policy = policy_params.copy()
        noised_policy += np.random.normal(0, noise, len(policy_params))
        noised_policy = np.clip(noised_policy, -1, 1)
        policy_p = noised_policy
    else:
        policy_p = policy_params
    #print(policy_params[0])

    param_policy.set_parameters(policy_p)
    outcome = run_episode(param_policy, distractors)
    a = round(outcome[3], 2)
    b = round(outcome[4], 2)
    if [a, b] != [round(-1.10355339, 2), round(0.60355339, 2)]:
        outs.append(1)
print(sum(outs))
#cp.disable()
#cp.dump_stats("test.cprof")
exit(0)


