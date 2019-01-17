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


def save_gep(gep, iteration, book_keeping, savefile_name, book_keeping_name, save_all=False):
    if save_all:
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
        #if render: arm_env.render()
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
arg_names = ['command', 'experiment_name', 'model_type', 'nb_iters', 'nb_bootstrap', 'explo_noise', 'distractors', 'interest_step']
args = dict(zip(arg_names, sys.argv))
Arg_list = collections.namedtuple('Arg_list', arg_names)
args = Arg_list(*(args.get(arg, None) for arg in arg_names))

exploration_noise = float(args.explo_noise) if args.explo_noise else 0.05
nb_bootstrap = int(args.nb_bootstrap) if args.nb_bootstrap else 1000
max_iterations = int(args.nb_iters) if args.nb_iters else 20000
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

# define variable's bounds for policy input and outcome
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

experiment_name = args.experiment_name if args.experiment_name else "experiment"
savefile_name = experiment_name + "_save.pickle"
book_keeping_file_name = experiment_name + "_bk.pickle"
save_step = 50000
save_all = False

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
# init IMGEP
full_outcome = input_names
full_outcome_bounds = b.get_bounds(full_outcome)

if model_type == "random_flat":
    outcome1 = full_outcome
    config = {'policy_nb_dims': total_policy_params,
              'modules': {'mod1': {'outcome_range': np.array([full_outcome.index(var) for var in outcome1])}}}
elif (model_type == "random_modular") or (args.model_type == "active_modular"):
    if not distractors:
        config = {'policy_nb_dims': total_policy_params,
                  'modules': {
                      'hand': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[0:3]])},
                      'stick1': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[3:5]])},
                      'stick2': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[5:7]])},
                      'magnet1': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[7:9]])},
                      'scratch1': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[9:11]])}}}
    else:
        config = {'policy_nb_dims': total_policy_params,
                  'modules': {'hand': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[0:3]])},
                              'stick1': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[3:5]])},
                              'stick2': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[5:7]])},
                              'magnet1': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[7:9]])},
                              'magnet2': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[9:11]])},
                              'magnet3': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[11:13]])},
                              'scratch1': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[13:15]])},
                              'scratch2': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[15:17]])},
                              'scratch3': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[17:19]])},
                              'cat': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[19:21]])},
                              'dog': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[21:23]])},
                              'static1': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[23:25]])},
                              'static2': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[25:27]])},
                              'static3': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[27:29]])},
                              'static4': {'outcome_range': np.array([full_outcome.index(var) for var in full_outcome[29:31]])}}}

    if model_type == "active_modular": model_babbling_mode = "active"
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
    if model_type == "active_modular":  # AMB init
        # active modules must perform an exploitation step periodically to compute interest for each modules
        interest_step = int(args.interest_step) if args.interest_step else 5

        gep = GEP(layers,
                  params,config,
                  model_babbling_mode="active",
                  explo_noise=exploration_noise,
                  update_interest_step=interest_step)
    else:  # F-RGB or RMB init
        gep = GEP(layers,params,config, model_babbling_mode="random", explo_noise=exploration_noise)

    # init boring book keeping
    b_k = dict()
    b_k['parameters'] = {'model_type': model_type,
                         'nb_bootstrap': nb_bootstrap,
                         'seed': seed,
                         'explo_noise': exploration_noise,
                         'distractors': distractors}
    if model_type == 'active_modular':
        b_k['parameters']['update_interest_step'] = interest_step

    for out in input_names:
        b_k['end_'+out] = []
    b_k['choosen_modules'] = []
    b_k['interests'] = dict()
    b_k['runtimes'] = {'produce': [], 'run': [], 'perceive': []}
    b_k['modules'] = {}

print("launching {}".format(b_k['parameters']))
#####################################################################
# init arm_env controller
arm_env = env=gym.make('ArmToolsToys-v0')
#arm_env.env.my_init(port=port, skip_step=4, tick_lengths=10)
for i in range(starting_iteration, max_iterations):
    if (i%100) == 0: print("{}: ########### Iteration # {} ##########".format(model_type, i))
    # generate policy using gep
    prod_time_start = time.time()
    policy_params = gep.produce(bootstrap=True) if i < nb_bootstrap else gep.produce()
    #print(policy_params.shape)
    prod_time_end = time.time()
    param_policy.set_parameters(policy_params)
    # if i > 2500:
    #     outcome = run_episode(param_policy, render = True)
    # else:
    outcome = run_episode(param_policy, distractors)
    #print(outcome)
    # a = round(outcome[3],2)
    # b = round(outcome[4],2)
    # if [a,b] != [round(-1.10355339,2), round(0.60355339,2)]:
    #     print('hello')
    #     with open("{}_policy_stick1_{}.pickle".format(experiment_name, time.time()), 'wb') as handle:
    #        pickle.dump(policy_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    run_ep_end = time.time()

    # scale outcome dimensions to [-1,1]
    scaled_outcome = scale_vector(outcome, np.array(full_outcome_bounds))
    gep.perceive(scaled_outcome)
    perceive_end = time.time()

    # boring book keeping
    b_k['runtimes']['produce'].append(prod_time_end - prod_time_start)
    b_k['runtimes']['run'].append(run_ep_end - prod_time_end)
    b_k['runtimes']['perceive'].append(perceive_end - run_ep_end)
    #print(b_k['runtimes']['produce'][-1])
    for out in input_names:
        b_k['end_'+out].append(outcome[full_outcome.index(out)])

    if ((i + 1) % save_step) == 0:
        print("saving gep")
        b_k['choosen_modules'] = gep.choosen_modules
        b_k['modules'] = gep.modules
        if model_type == "active_modular":
            b_k['interests'] = gep.interests
        save_gep(gep, i + 1, b_k, savefile_name, book_keeping_file_name, save_all)
print("closing {}".format(b_k['parameters']))
#cp.disable()
#cp.dump_stats("test.cprof")
exit(0)


