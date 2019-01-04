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
from utils.nn_policy import Simple_NN
import matplotlib.pyplot as plt
from utils.plot_utils import *
import collections
from collections import OrderedDict
from utils.gep_utils import *
# from malmo_controller import MalmoController
import gym2
import config


def get_outcome(state):
    outcome = state.tolist()
    if distractors:
        # add 2 moving and 2 fixed distractors to the outcome space
        fixed_distractor1 = [-0.7, 0.3, 0.5]
        fixed_distractor2 = [0.1, 0.2, -0.5]
        moving_distractor1 = np.random.random(3) * 2 - 1
        moving_distractor2 = np.random.random(3) * 2 - 1
        distractors_final_state = fixed_distractor1 + moving_distractor1.tolist() + fixed_distractor2 + moving_distractor2.tolist()
        outcome += distractors_final_state
    return np.array(outcome)

def run_episode(policy_params, add_noise=False, noise=0.025):
    print(policy_params[0])
    if add_noise:
        noised_policy = policy_params.copy()
        noised_policy += np.random.normal(0, noise, len(policy_params))
        noised_policy = np.clip(noised_policy, -1, 1)
        policy_params = noised_policy
    print(policy_params[0])
    out = malmo.reset()
    state = out['observation']
    # Loop until mission/episode ends:
    done = False
    while not done:
        # extract the world state that will be given to the agent's policy
        actions = param_policy.forward(state.reshape(1, -1), policy_params)
        out, _, done, _ = malmo.step(actions)
        state = out['observation']
    return get_outcome(state)


# define and parse argument values
# more info here: https://stackoverflow.com/questions/5423381/checking-if-sys-argvx-is-defined
arg_names = ['command', 'experiment_name', 'model_type', 'nb_iters', 'nb_bootstrap', 'explo_noise', 'server_port',
             'distractors', 'interest_step']
args = dict(zip(arg_names, sys.argv))
Arg_list = collections.namedtuple('Arg_list', arg_names)
args = Arg_list(*(args.get(arg, None) for arg in arg_names))

exploration_noise = float(args.explo_noise) if args.explo_noise else 0.10
nb_bootstrap = int(args.nb_bootstrap) if args.nb_bootstrap else 1000
max_iterations = int(args.nb_iters) if args.nb_iters else 10000
# possible models: ["random_modular", "random_flat", "active_modular"]
model_type = args.model_type if args.model_type else "random_modular"
if args.distractors:
    if args.distractors == 'True':
        distractors = True
        distr_names = ['fixed_distr_x1', 'fixed_distr_y1', 'fixed_distr_z1', 'moving_distr_x1', 'moving_distr_y1',
                       'moving_distr_z1',
                       'fixed_distr_x2', 'fixed_distr_y2', 'fixed_distr_z2', 'moving_distr_x2', 'moving_distr_y2',
                       'moving_distr_z2']
    elif args.distractors == 'False':
        distractors = False
    else:
        print('distractor option not recognized, choose True or False')
        raise NameError
else:
    distractors = False

# environment-related init
nb_blocks = 5
# define variable's bounds for policy input and outcome
state_names = ['agent_x', 'agent_z', 'pickaxe_x', 'pickaxe_z', 'shovel_x', 'shovel_z'] + \
              ['block_' + str(i) for i in range(nb_blocks)] + ['cart_x']
b = config.get_env_bounds('emmc_env')
if distractors:
    for d_name in distr_names:
        b.add(d_name, [-1, 1])

experiment_name = args.experiment_name if args.experiment_name else "experiment"
savefile_name = experiment_name + "_save.pickle"
book_keeping_file_name = experiment_name + "_bk.pickle"
save_step = 1000

# init neural network policy
input_names = state_names
input_bounds = b.get_bounds(input_names)
input_size = len(input_bounds)
print('input_bounds: %s' % input_bounds)
hidden_layer_size = 64
action_set_size = 3
param_policy = Simple_NN(input_size, input_bounds, action_set_size, hidden_layer_size)
total_policy_params = hidden_layer_size * input_size + hidden_layer_size * action_set_size

#####################################################################
port = int(args.server_port) if args.server_port else None
# init malmo controller
malmo = gym2.make('ExtendedMalmoMountainCart-v0')
malmo.env.my_init(port=port, skip_step=4, tick_lengths=25)

with open("emmccpu_amb_0_policy_cart_1546507987.0211735.pickle", 'rb') as handle:
    policy_params = pickle.load(handle)
outs=[]
for i in range(0, 100):
    time.sleep(0.5)
    print("########### Iteration # %s ##########" % (i))
    outcome = run_episode(policy_params, add_noise=True)
    print(outcome)
    if outcome[-1] != 291.5:
        outs.append(outcome[-1])
print(outs)
print(len(outs))

exit(0)
