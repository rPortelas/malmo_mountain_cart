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
arg_names = ['command', 'experiment_name', 'model_type', 'nb_iters', 'nb_bootstrap', 'explo_noise', 'server_port',
             'distractors', 'trajectories', 'interest_step']
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
    elif args.distractors == 'False':
        distractors = False
    else:
        print('distractor option not recognized, choose True or False')
        raise NameError
else:
    distractors = False

if args.trajectories:
    if args.trajectories == 'True':
        trajectories = True
    elif args.trajectories == 'False':
        trajectories = False
    else:
        print('trajectory option not recognized, choose True or False')
        raise NameError
else:
    trajectories = False

if trajectories:
    nb_traj_steps = 5
else:
    nb_traj_steps = 1

# environment-related init
nb_blocks = 5
# define variable's bounds for policy input and outcome
state_names = ['agent_x', 'agent_z', 'pickaxe_x', 'pickaxe_z', 'shovel_x', 'shovel_z'] + \
              ['block_' + str(i) for i in range(nb_blocks)] + ['cart_x']

if distractors:
    nb_distractors = 3
    Distractor_simulator = Distractors(nb_distractors=nb_distractors, noise=0.0) # fixed distractors
    for i in range(nb_distractors):
        state_names.append('dist'+str(i)+'_x')
        state_names.append('dist'+str(i)+'_y')

b = conf.get_env_bounds('emmc_env')

experiment_name = args.experiment_name if args.experiment_name else "experiment"
savefile_name = experiment_name + "_save.pickle"
book_keeping_file_name = experiment_name + "_bk.pickle"
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

if (model_type == "random_flat") or (model_type == "random"):
    outcome1 = full_outcome
    config = {'policy_nb_dims': total_policy_params,
              'modules': {'mod1': {'outcome_range': np.arange(0,len(full_outcome),1),
                                   'focus_state_range': np.arange(0,len(full_outcome),1)//nb_traj_steps}}}
elif model_type == "single_goal_space":
    nb_t = nb_traj_steps
    config = {'policy_nb_dims': total_policy_params}
    config['modules'] = {}
    for names, inds in zip(objects, objects_idx):
        mod_name = names[0][:-2]
        if mod_name == "cart":
            start_idx = inds[0] * nb_traj_steps
            end_idx = inds[-1] * nb_traj_steps
            config['modules'][mod_name] = {}
            config['modules'][mod_name]['outcome_range'] = np.arange(start_idx, end_idx, 1)
            config['modules'][mod_name]['focus_state_range'] = np.arange(inds[0], inds[-1], 1)
elif (model_type == "random_modular") or (args.model_type == "active_modular"):
    nb_t = nb_traj_steps
    config = {'policy_nb_dims': total_policy_params}
    config['modules'] = {}
    for names,inds in zip(objects, objects_idx):
        mod_name = names[0][:-2]
        start_idx = inds[0] * nb_traj_steps
        end_idx = inds[-1] * nb_traj_steps
        config['modules'][mod_name] = {}
        config['modules'][mod_name]['outcome_range'] = np.arange(start_idx, end_idx, 1)
        config['modules'][mod_name]['focus_state_range'] = np.arange(inds[0], inds[-1], 1)
    if model_type == "active_modular": model_babbling_mode = "active"
else:
    raise NotImplementedError
print(config)
# if a gep save exist, load gep, init it otherwise
if os.path.isfile(savefile_name):
    gep, starting_iteration, b_k = load_gep(savefile_name, book_keeping_file_name)
    nb_bootstrap = b_k['parameters']['nb_bootstrap']
    np.random.seed(b_k['parameters']['seed'])

else:
    seed = np.random.randint(1000)
    np.random.seed(seed)
    if model_type == "active_modular":  # AMB init
        interest_mean_rate=100.
        # active modules must perform an exploitation step periodically to compute interest for each modules
        interest_step = int(args.interest_step) if args.interest_step else 5

        gep = GEP(layers,
                  params,
                  config,
                  model_babbling_mode="active",
                  explo_noise=exploration_noise,
                  interest_mean_rate=interest_mean_rate,
                  update_interest_step=interest_step)
    else:  # F-RGB or RMB init
        gep = GEP(layers, params, config, model_babbling_mode="random", explo_noise=exploration_noise)

    starting_iteration = 0
    # init boring book keeping
    b_k = dict()
    b_k['parameters'] = {'model_type': model_type,
                         'nb_bootstrap': nb_bootstrap,
                         'seed': seed,
                         'explo_noise': exploration_noise,
                         'distractors': distractors,
                         'trajectories': trajectories,
                         'size_sequential_nn': size_sequential_nn}
    if model_type == 'active_modular':
        b_k['parameters']['update_interest_step'] = interest_step
        b_k['parameters']['interest_mean_rate'] = interest_mean_rate

    b_k['choosen_modules'] = []
    b_k['interests'] = dict()
    b_k['runtimes'] = {'produce': [], 'run': [], 'perceive': []}
    b_k['modules'] = {}

    for out in input_names:
        b_k['end_'+out] = []

print("launching {}".format(b_k['parameters']))
#####################################################################
port = int(args.server_port) if args.server_port else None
# init env controller
env = gym2.make('ExtendedMalmoMountainCart-v0')
env.env.my_init(port=port, skip_step=4, tick_lengths=15, desired_mission_time=10)

for i in range(starting_iteration, max_iterations):
    print("########### Iteration # %s ##########" % (i))
    # generate policy using gep
    prod_time_start = time.time()
    policy_params, focus, add_noise = gep.produce(bootstrap=True) if i < nb_bootstrap else gep.produce()
    prod_time_end = time.time()
    outcome, states = run_episode(model_type, param_policy, policy_params, exploration_noise, distractors, nb_traj_steps, size_sequential_nn, focus_range=focus, add_noise=add_noise)
    run_ep_end = time.time()
    # scale outcome dimensions to [-1,1]
    scaled_outcome = scale_vector(outcome, np.array(full_outcome_bounds))
    gep.perceive(scaled_outcome, policy_params)
    perceive_end = time.time()
    # if not outcome[-1] == 291.5:
    #     with open("{}_gepexplore_p_cart2_{}.pickle".format(experiment_name, time.time()), 'wb') as handle:
    #         pickle.dump(policy_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # boring book keeping
    b_k['runtimes']['produce'].append(prod_time_end - prod_time_start)
    b_k['runtimes']['run'].append(run_ep_end - prod_time_end)
    b_k['runtimes']['perceive'].append(perceive_end - run_ep_end)
    #print(b_k['runtimes']['produce'][-1])
    for out in input_names:
        b_k['end_'+out].append(states[-1][input_names.index(out)])

    if ((i + 1) % save_step) == 0:
        print("saving gep")
        b_k['choosen_modules'] = gep.choosen_modules
        #b_k['modules'] = gep.modules
        if model_type == "active_modular":
            b_k['interests'] = gep.interests
        save_gep(gep, i + 1, b_k, savefile_name, book_keeping_file_name)
print("closing {}".format(b_k['parameters']))
exit(0)