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
from utils.neural_network import PolicyNN
# from malmo_controller import MalmoController
import gym2
import config as conf

def get_outcome(states, distractors, nb_traj_steps):
    #print(len(states))
    step_size = (len(states)-1)//nb_traj_steps
    steps = np.arange(step_size,len(states),step_size)
    outcome = []
    start = 0
    for idx in objects_idx:
        for step in steps:
            s = states[step].tolist()
            outcome += s[idx[0]:idx[1]]
    return outcome


def get_state(state, distractors):
    s = state
    if distractors:
        Distractor_simulator.step()
        d_arr = Distractor_simulator.get()
        s = np.array(s.tolist() + d_arr.tolist())
    return np.array(s)

def save_gep(gep, iteration, book_keeping, savefile_name, book_keeping_name):
    gep.prepare_pickling()
    with open(savefile_name, 'wb') as handle:
        pickle.dump([gep, iteration], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(book_keeping_name, 'wb') as handle:
        pickle.dump(book_keeping, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_gep(savefile_name, book_keeping_name):
    with open(savefile_name, 'rb') as handle:
        gep, starting_iteration = pickle.load(handle)
    with open(book_keeping_name, 'rb') as handle:
        b_k = pickle.load(handle)
    return gep, starting_iteration, b_k




# def run_episode(model):
#     out = malmo.reset()
#     state = out['observation']
#     # Loop until mission/episode ends:
#     # Loop until mission/episode ends
#     done = False
#     while not done:
#         # extract the world state that will be given to the agent's policy
#         normalized_state = scale_vector(state, np.array(input_bounds))
#         actions = model.get_action(normalized_state.reshape(1, -1))
#         out, _, done, _ = malmo.step(actions[0])
#         state = out['observation']
#     return get_outcome(state)

def run_episode(model_type, model, policy_params, explo_noise, distractors, nb_traj_steps, size_sequential_nn, max_step=40, focus_range=None, add_noise=False):
    if distractors: Distractor_simulator.reset()
    out = env.reset()
    state = get_state(out['observation'], distractors)
    if add_noise:
        #print(state)
        init_focus_state = np.array([state[i] for i in focus_range])
        #print(init_focus_state)
    # Loop until mission/episode ends:
    done = False
    states = [state]
    steps_per_nn = int(max_step / len(policy_params))
    #print("steps per nn {}".format(steps_per_nn))
    while not done:
        for nn_idx in range(len(policy_params)):
            if add_noise:
                if (not ([state[i] for i in focus_range] == init_focus_state).all()) or (size_sequential_nn == 1) or (model_type == 'random_flat'):
                    #object of interest moved during previous neural net, lets add noise for the following nets
                    policy_params[nn_idx] += np.random.normal(0, explo_noise, len(policy_params[nn_idx]))
                    policy_params[nn_idx] = np.clip(policy_params[nn_idx], -1, 1)
                    #policy_params[nn_idx] = get_random_nn(layers, params)
            model.set_parameters(policy_params[nn_idx])
            for i in range(steps_per_nn):
                # extract the world state that will be given to the agent's policy
                normalized_state = scale_vector(state, np.array(input_bounds))
                actions = model.get_action(normalized_state.reshape(1, -1))
                print(actions)
                out, _, done, _ = env.step(actions[0])
                #env.render()
                state = get_state(out['observation'], distractors)
                states.append(state)
    assert(done)
    return get_outcome(states, distractors, nb_traj_steps), states


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# define and parse argument values
# more info here: https://stackoverflow.com/questions/5423381/checking-if-sys-argvx-is-defined
arg_names = ['command', 'experiment_name', 'server_port']
args = dict(zip(arg_names, sys.argv))
Arg_list = collections.namedtuple('Arg_list', arg_names)
args = Arg_list(*(args.get(arg, None) for arg in arg_names))

# environment-related init
nb_blocks = 5
# define variable's bounds for policy input and outcome
state_names = ['agent_x', 'agent_z', 'pickaxe_x', 'pickaxe_z', 'shovel_x', 'shovel_z'] + \
              ['block_' + str(i) for i in range(nb_blocks)] + ['cart_x']
distractors = True
if distractors:
    nb_distractors = 3
    Distractor_simulator = Distractors(nb_distractors=nb_distractors, noise=0.0) # fixed distractors
    for i in range(nb_distractors):
        state_names.append('dist'+str(i)+'_x')
        state_names.append('dist'+str(i)+'_y')

b = conf.get_env_bounds('emmc_env')

experiment_name = args.experiment_name if args.experiment_name else "experiment"
savefile_name = '' + experiment_name + "_save.pickle"
book_keeping_file_name = '' + experiment_name + "_bk.pickle"
save_step = 1000

# init neural network policy
size_sequential_nn = 5

input_names = state_names
input_bounds = b.get_bounds(input_names)
input_size = len(input_bounds)
print('input_bounds: %s' % input_bounds)
layers = [64]
action_set_size = 3
params = {'layers': layers, 'activation_function':'relu', 'max_a':1.,
          'dims':{'s':input_size,'a':action_set_size},'bias':True, 'size_sequential_nn':size_sequential_nn}
param_policy = PolicyNN(params)
total_policy_params = get_n_params(param_policy)

# if a gep save exist, load gep, init it otherwise
if os.path.isfile(savefile_name):
    gep, starting_iteration, b_k = load_gep(savefile_name, book_keeping_file_name)
    print(b_k['parameters'])
    nb_bootstrap = b_k['parameters']['nb_bootstrap']
    seed = b_k['parameters']['seed']
    model_type = b_k['parameters']['model_type']
    exploration_noise = b_k['parameters']['explo_noise']
    if b_k['parameters']['trajectories'] == False:
        nb_traj_steps = 1
    np.random.seed(seed)
else:
    print('FILE IS NOT DETECTED: {}'.format(savefile_name))
    raise NotImplementedError

# init IMGEP
full_outcome = input_names
objects, objects_idx = conf.get_objects('emmc_env')
if distractors:
    for i in range(nb_distractors):
        objects.append(['dist'+str(i)+'_x', 'dist'+str(i)+'_y'])
        prev_idx = objects_idx[-1][1]
        objects_idx.append([prev_idx, prev_idx + 2])
full_outcome = []
for obj in objects:
    full_outcome += obj * nb_traj_steps
full_outcome_bounds = b.get_bounds(full_outcome)


print("launching {}".format(b_k['parameters']))
#####################################################################
port = int(args.server_port) if args.server_port else None
# init env controller
env = gym2.make('ExtendedMalmoMountainCart-v0')
env.env.my_init(port=port, skip_step=4, tick_lengths=50, desired_mission_time=10)
test_cart = True
test_pickaxe = False
cart_states = True
# CART GOALS
step = np.abs((-0.95 - 0.78) /100) # goals are sampled only on reachable space (the cart goal space was loosely defined
#cart_goals = np.arange(-0.95,0.78,step)
cart_goals = [0.468599999]
cart_errors = []
cart_outcomes = []
cart_states = []
nb_retry = 10
#[-0.8635,-0.48290000000000044,-0.10230000000000083,0.4685999999999986,0.693499999999998220]
if test_cart:
    for i,g in enumerate(cart_goals):
        # generate policy using gep
        policy_params, focus, add_noise = gep.produce(normalized_goal=[g], goal_space_name='cart')
        batch_errors = []
        batch_outcomes = []
        for j in range(nb_retry):
            outcome, states = run_episode(model_type, param_policy, policy_params, exploration_noise, distractors, nb_traj_steps, size_sequential_nn, focus_range=None, add_noise=False)
            cart_states.append(states)
            # scale outcome dimensions to [-1,1]
            scaled_outcome = scale_vector(outcome, np.array(full_outcome_bounds))
            cart_outcome = scaled_outcome[11]
            batch_outcomes.append(cart_outcome)
            error = np.abs(g - cart_outcome)
            batch_errors.append(error)
        cart_outcomes.append(batch_outcomes)
        cart_errors.append(batch_errors)
        print("{}: goal was {}, outcome {}, error = {}".format(i, g, batch_outcomes, batch_errors))

if test_pickaxe:
    # PICKAXE GOALS
    pickaxe_goals_x = np.arange(-1.,1.,0.1)
    pickaxe_goals_z = np.arange(-1.,1.,0.05)
    #pickaxe_goals_x = np.arange(-1.,1.,0.5)
    #pickaxe_goals_z = np.arange(-1.,1.,0.5)
    # for g in goals:
    #     print(scale_vector(np.array(g), b.get_bounds(['cart_x'])))
    # exit(0)
    pickaxe_errors = []
    pickaxe_outcomes = []
    pickaxe_goals_2d = []
    for i,g_x in enumerate(pickaxe_goals_x):
        for j,g_z in enumerate(pickaxe_goals_z):
            g = [g_x,g_z]
            pickaxe_goals_2d.append(g)
            # generate policy using gep
            prod_time_start = time.time()
            policy_params, focus, add_noise = gep.produce(normalized_goal=g, goal_space_name='pickaxe')
            prod_time_end = time.time()
            outcome, states = run_episode(model_type, param_policy, policy_params, exploration_noise, distractors, nb_traj_steps, size_sequential_nn, focus_range=None, add_noise=False)
            run_ep_end = time.time()
            # scale outcome dimensions to [-1,1]
            scaled_outcome = scale_vector(outcome, np.array(full_outcome_bounds))
            #print(outcome)
            pickaxe_outcome = scaled_outcome[2:4]
            pickaxe_outcomes.append(pickaxe_outcome)
            error = np.abs(g - pickaxe_outcome)
            unsc_g = unscale_vector(np.array(g), np.array(b.get_bounds(['pickaxe_x', 'pickaxe_z'])))
            print("{}: goal was {} ({}), outcome {}, error = {}".format(i, g, unsc_g, pickaxe_outcome, error))
            pickaxe_errors.append(error)
            perceive_end = time.time()


test_data = {}
if test_cart:
    test_data['cart_goals'] = cart_goals
    test_data['cart_outcomes'] = cart_outcomes
    test_data['cart_errors'] = cart_errors
    test_data['cart_states'] = cart_states
if test_pickaxe:
    test_data['pickaxe_goals'] = pickaxe_goals_2d
    test_data['pickaxe_outcomes'] = pickaxe_outcomes
    test_data['pickaxe_errors'] = pickaxe_errors
pickle.dump(test_data, open(experiment_name+"_testwithstates.pickle", "wb" ))
exit(0)