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
#from malmo_controller import MalmoController
import gym2
import config

def get_outcome(state):
    outcome = state.tolist()
    if distractors:
        #add 2 moving and 2 fixed distractors to the outcome space
        fixed_distractor1 = [-0.7,0.3,0.5]
        fixed_distractor2 = [0.1,0.2,-0.5]
        moving_distractor1 = np.random.random(3) * 2 - 1
        moving_distractor2 = np.random.random(3) * 2 - 1
        distractors_final_state = fixed_distractor1 + moving_distractor1.tolist() + fixed_distractor2 + moving_distractor2.tolist()
        outcome += distractors_final_state
    return np.array(outcome)


def save_gep(gep, iteration, book_keeping, savefile_name, book_keeping_name):
    with open(savefile_name, 'wb') as handle:
        pickle.dump([gep, iteration], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(book_keeping_name, 'wb') as handle:
        pickle.dump(book_keeping, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_gep(savefile_name, book_keeping_name):
    with open(savefile_name, 'rb') as handle:
        gep,starting_iteration = pickle.load(handle)
    with open(book_keeping_name, 'rb') as handle:
        b_k = pickle.load(handle)
    return gep, starting_iteration, b_k

def run_episode(policy_params):
    out = malmo.reset()
    state = out['observation']
    # Loop until mission/episode ends:
    done = False
    while not done:
        # extract the world state that will be given to the agent's policy
        actions = param_policy.forward(state.reshape(1,-1), policy_params)
        out,_, done, _ = malmo.step(actions)
        state = out['observation']
    return get_outcome(state)

# define and parse argument values
# more info here: https://stackoverflow.com/questions/5423381/checking-if-sys-argvx-is-defined
arg_names = ['command','experiment_name','model_type','nb_iters','nb_bootstrap','explo_noise','server_port','distractors','interest_step']
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
        distr_names = ['fixed_distr_x1', 'fixed_distr_y1', 'fixed_distr_z1', 'moving_distr_x1', 'moving_distr_y1', 'moving_distr_z1',
                       'fixed_distr_x2', 'fixed_distr_y2', 'fixed_distr_z2', 'moving_distr_x2', 'moving_distr_y2', 'moving_distr_z2']
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
state_names = ['agent_x', 'agent_z', 'pickaxe_x', 'pickaxe_z', 'shovel_x', 'shovel_z'] +\
              ['block_' + str(i) for i in range(nb_blocks)] + ['cart_x']
b = config.get_env_bounds('emmc_env')
if distractors:
    for d_name in distr_names:
       b.add(d_name, [-1,1]) 

experiment_name = args.experiment_name if args.experiment_name else "experiment"
savefile_name = experiment_name+"_save.pickle"
book_keeping_file_name = experiment_name+"_bk.pickle"
save_step = 1000

# init neural network policy
input_names = state_names
input_bounds = b.get_bounds(input_names)
input_size = len(input_bounds)
print('input_bounds: %s' % input_bounds) 
hidden_layer_size = 64
action_set_size = 3
param_policy = Simple_NN(input_size, input_bounds, action_set_size , hidden_layer_size)
total_policy_params = hidden_layer_size*input_size + hidden_layer_size*action_set_size

# init IMGEP
full_outcome = input_names
if distractors: full_outcome += distr_names
full_outcome_bounds = b.get_bounds(full_outcome)

if model_type == "random_flat":
    outcome1 = full_outcome
    config = {'policy_nb_dims': total_policy_params,
              'modules':{'mod1':{'outcome_range': np.array([full_outcome.index(var) for var in outcome1])}}}
elif (model_type == "random_modular") or (args.model_type == "active_modular"):
    agent_xz = full_outcome[:2]
    pickaxe_xz = full_outcome[2:4]
    shovel_xz = full_outcome[4:6]
    blocks = full_outcome[6:11]
    cart_x = [full_outcome[11]]
    config = {'policy_nb_dims': total_policy_params,
              'modules':{'agent_end_pos':{'outcome_range': np.array([full_outcome.index(var) for var in agent_xz])},
                         'pickaxe_end_pos':{'outcome_range': np.array([full_outcome.index(var) for var in pickaxe_xz])},
                         'shovel_end_pos':{'outcome_range':np.array([full_outcome.index(var) for var in shovel_xz])},
                         'mined_blocks': {'outcome_range': np.array([full_outcome.index(var) for var in blocks])},
                         'cart_end_pos': {'outcome_range': np.array([full_outcome.index(var) for var in cart_x])}}}
    if distractors:
        fixed_distr_outcomes = ['fixed_distr_x', 'fixed_distr_y', 'fixed_distr_z']
        moving_distr_outcomes = ['moving_distr_x', 'moving_distr_y', 'moving_distr_z']
        config['modules']['fixed_dist_final_pos1'] = {'outcome_range': np.array([full_outcome.index(var+'1') for var in fixed_distr_outcomes])}
        config['modules']['moving_dist_final_pos1'] = {'outcome_range': np.array([full_outcome.index(var+'2') for var in moving_distr_outcomes])}
        config['modules']['fixed_dist_final_pos2'] = {'outcome_range': np.array([full_outcome.index(var+'1') for var in fixed_distr_outcomes])}
        config['modules']['moving_dist_final_pos2'] = {'outcome_range': np.array([full_outcome.index(var+'2') for var in moving_distr_outcomes])}

    if model_type == "active_modular": model_babbling_mode ="active"
else:
    raise NotImplementedError


# if a gep save exist, load gep, init it otherwise
if os.path.isfile(savefile_name):
    gep, starting_iteration, b_k = load_gep(savefile_name, book_keeping_file_name)
    nb_bootstrap = b_k['parameters']['nb_bootstrap']
    np.random.seed(b_k['parameters']['seed'])

else:
    starting_iteration = 0
    seed = np.random.randint(1000)
    np.random.seed(seed)
    if model_type == "active_modular": # AMB init
        # active modules must perform an exploitation step periodically to compute interest for each modules
        interest_step = int(args.interest_step) if args.interest_step else 5

        gep = GEP(config,
                  model_babbling_mode="active", 
                  explo_noise=exploration_noise, 
                  update_interest_step= interest_step)
    else: # F-RGB or RMB init
        gep = GEP(config, model_babbling_mode="random", explo_noise=exploration_noise)

    # init boring book keeping
    b_k = dict()
    b_k['parameters'] = {'model_type': model_type,
                         'nb_bootstrap': nb_bootstrap,
                         'seed': seed,
                         'explo_noise': exploration_noise,
                         'distractors': distractors}
    if model_type == 'active_modular':
        b_k['parameters']['update_interest_step'] = interest_step
    b_k['end_agent_x'] = []
    b_k['end_agent_z'] = []
    b_k['end_pickaxe_x'] = []
    b_k['end_pickaxe_z'] = []
    b_k['end_shovel_x'] = []
    b_k['end_shovel_z'] = []
    b_k['end_cart_x'] = []
    b_k['choosen_modules'] = []
    b_k['interests'] = dict()
    b_k['runtimes'] = {'produce':[], 'run':[], 'perceive':[]}
    for i in range(nb_blocks):
            b_k['end_block_'+str(i)] = []

print("launching {}".format(b_k['parameters']))
#####################################################################
port = int(args.server_port) if args.server_port else None
# init malmo controller
malmo = gym2.make('ExtendedMalmoMountainCart-v0')
malmo.env.my_init(port=port, skip_step=4, tick_lengths=10)

for i in range(starting_iteration,max_iterations):
    print("########### Iteration # %s ##########" % (i))
    # generate policy using gep
    prod_time_start = time.time()
    policy_params = gep.produce(bootstrap=True) if i < nb_bootstrap else gep.produce()
    prod_time_end = time.time()
    outcome = run_episode(policy_params)
    run_ep_end = time.time()

    # scale outcome dimensions to [-1,1]
    scaled_outcome = scale_vector(outcome, np.array(full_outcome_bounds))
    gep.perceive(scaled_outcome)
    perceive_end = time.time()


    # boring book keeping
    b_k['runtimes']['produce'].append(prod_time_end - prod_time_start)
    b_k['runtimes']['run'].append(run_ep_end - prod_time_end)
    b_k['runtimes']['perceive'].append(perceive_end - run_ep_end)
    b_k['end_agent_x'].append(outcome[full_outcome.index('agent_x')])
    b_k['end_agent_z'].append(outcome[full_outcome.index('agent_z')])
    b_k['end_pickaxe_x'].append(outcome[full_outcome.index('pickaxe_x')])
    b_k['end_pickaxe_z'].append(outcome[full_outcome.index('pickaxe_z')])
    b_k['end_shovel_x'].append(outcome[full_outcome.index('shovel_x')])
    b_k['end_shovel_z'].append(outcome[full_outcome.index('shovel_z')])
    b_k['end_cart_x'].append(outcome[full_outcome.index('cart_x')])
    for k in range(nb_blocks):
        b_k['end_block_'+str(k)].append(outcome[full_outcome.index('block_'+str(k))])
    
    if ((i+1) % save_step) == 0:
        print("saving gep")
        b_k['choosen_modules'] = gep.choosen_modules
        if model_type == "active_modular":
            b_k['interests'] = gep.interests
        save_gep(gep, i+1, b_k, savefile_name, book_keeping_file_name)
print("closing {}".format(b_k['parameters']))
exit(0)
