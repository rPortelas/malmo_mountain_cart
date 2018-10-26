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


# evaluate model over given goals in [-1,1], returns errors for each sub space
def eval(agent_pos_goals, cart_x_goals, breads_goals):
    agent_goals_range = range(len(agent_pos_goals))
    cart_goals_range = range(len(agent_pos_goals),len(agent_pos_goals)+len(cart_x_goals))
    breads_goals_range = range(len(agent_pos_goals)+len(cart_x_goals),len(agent_pos_goals)+len(cart_x_goals)+len(breads_goals))
    all_goals = agent_pos_goals.tolist() + cart_x_goals.tolist() + breads_goals.tolist()
    #print(all_goals)
    agent_errors = []
    cart_errors = []
    breads_errors = []
    cart_touched = []

    for i,goal in enumerate(all_goals):
        print("########### Evaluation Iteration # %s ##########" % (i))
        goal = np.array(goal)
        # generate exploitation policy using gep (NN exploitation)
        goal_range = None
        if i in agent_goals_range:
            #print('agent')
            goal_range = [0,1,2]
        elif i in cart_goals_range:
            #print('cart')
            goal_range = [3]
        elif i in breads_goals_range:
            #print('breads')
            goal_range = [4]
        else:
            raise NotImplementedError
        policy_params = gep.produce(normalized_goal=goal,goal_range=goal_range)
        outcome = run_episode(policy_params)
        #print('result: %s' % outcome[goal_range])
        # normalize outcome
        outcome = scale_vector(outcome, np.array(full_outcome_bounds))

        #print('goal was {}'.format(unscale_vector(goal, np.array(full_outcome_bounds)[goal_range])))
        sub_outcome = outcome[goal_range]
        error = np.abs(sub_outcome - goal)

        # add error
        if i in agent_goals_range:
            agent_errors.append(np.sum(error) / 3.)
        elif i in cart_goals_range:
            cart_errors.append(error[0])
            #print(sub_outcome)
            if sub_outcome != scale_vector([291.5],np.array(b.get_bounds(['cart_x']))):
                #print('touched')
                cart_touched.append(1)
            else:
                #print('not touched')
                cart_touched.append(0)
        elif i in breads_goals_range:
            breads_errors.append(error[0])
        else:
            raise NotImplementedError
    return np.mean(agent_errors), np.mean(cart_errors), np.mean(breads_errors), cart_touched

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
nb_breads = 5
# define variable's bounds for policy input and outcome 
b = Bounds()
b.add('agent_x',[288.3,294.7])
b.add('agent_y',[4,6])
b.add('agent_z',[433.3,443.7])
b.add('cart_x',[285,297])
for i in range(nb_breads):
    b.add('bread_'+str(i),[0,1])
# add meta variable
b.add('breads',[0,nb_breads])
if distractors:
    for d_name in distr_names:
       b.add(d_name, [-1,1]) 

experiment_name = args.experiment_name if args.experiment_name else "experiment"
savefile_name = experiment_name+"_save.pickle"
book_keeping_file_name = experiment_name+"_bk.pickle"
save_step = 200
plot_step = 100000
#eval_step = 200

# init neural network policy
input_names = ['agent_x','agent_y','agent_z','cart_x'] + ['bread_'+str(i) for i in range(nb_breads)]
input_bounds = b.get_bounds(input_names)
input_size = len(input_bounds)
print('input_bounds: %s' % input_bounds) 
hidden_layer_size = 64
action_set_size = 2 
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
    outcome1 = full_outcome[:3]
    outcome2 = [full_outcome[3]]
    outcome3 = full_outcome[4:9]
    config = {'policy_nb_dims': total_policy_params,
              'modules':{'agent_final_pos':{'outcome_range': np.array([full_outcome.index(var) for var in outcome1])},
                         'cart_final_pos':{'outcome_range': np.array([full_outcome.index(var) for var in outcome2])},
                         'bread_final_count':{'outcome_range':np.array([full_outcome.index(var) for var in outcome3])}}}
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
        gep = GEP(config,model_babbling_mode="random", explo_noise=exploration_noise)

    # init boring book keeping
    b_k = dict()
    b_k['parameters'] = {'model_type': model_type,
                         'nb_bootstrap': nb_bootstrap,
                         'seed': seed,
                         'explo_noise': exploration_noise,
                         'distractors': distractors}
    if model_type == 'active_modular':
        b_k['parameters']['update_interest_step'] = interest_step
    b_k['final_agent_x_reached'] = []
    b_k['final_agent_z_reached'] = []
    b_k['final_cart_x_reached'] = []
    b_k['final_bread_recovered'] = []
    b_k['choosen_modules'] = []
    b_k['interests'] = dict()
    for i in range(nb_breads):
            b_k['bread_'+str(i)] = []
    #b_k['eval_errors'] = None

'''
# load test set goals
if os.path.isfile('test_set_goals.pickle'):
    with open('test_set_goals.pickle', 'rb') as handle:
        agent_pos_goals, cart_x_goals, breads_goals = pickle.load(handle)
else:
    print("TEST SET NOT FOUND: run generate_random_goals script first")
'''
#####################################################################

port = int(args.server_port) if args.server_port else 10000
# init malmo controller
#malmo = MalmoController(port=port, tick_lengths=15)
malmo = gym2.make('MalmoMountainCart-v0')
malmo.env.my_init(skip_step=4, tick_lengths=10)

for i in range(starting_iteration,max_iterations):
    print("########### Iteration # %s ##########" % (i))
    # generate policy using gep
    policy_params = gep.produce(bootstrap=True) if i < nb_bootstrap else gep.produce()
    outcome = run_episode(policy_params)

    # scale outcome dimensions to [-1,1]
    scaled_outcome = scale_vector(outcome, np.array(full_outcome_bounds))
    gep.perceive(scaled_outcome)

    # boring book keeping
    b_k['final_agent_x_reached'].append(outcome[full_outcome.index('agent_x')])
    b_k['final_agent_z_reached'].append(outcome[full_outcome.index('agent_z')])
    b_k['final_cart_x_reached'].append(outcome[full_outcome.index('cart_x')])
    b_k['final_bread_recovered'].append(int(sum(outcome[4:9])))
    for k in range(nb_breads):
        b_k['bread_'+str(k)].append(outcome[full_outcome.index('bread_'+str(k))])
    
    if ((i+1) % save_step) == 0:
        print("saving gep")
        b_k['choosen_modules'] = gep.choosen_modules
        if model_type == "active_modular":
            b_k['interests'] = gep.interests
        save_gep(gep, i+1, b_k, savefile_name, book_keeping_file_name)
    
    '''
    if ((i+1) % plot_step) == 0:
        print("plotting")
        #plot_agent_pos_exploration(1, b_k['final_agent_x_reached'], b_k['final_agent_z_reached'])
        #plot_agent_cart_exploration(2, b_k['final_cart_x_reached'])
        #plot_agent_bread_exploration(3, b_k['final_bread_recovered'])
        if model_type == "active_modular":
            plot_interests(5, gep.interests)
        plt.show(block=False)
    '''

# offline, in depth competence error evaluation
# load random dataset of goals
'''
if os.path.isfile('test_set_goals.pickle'):
    with open('large_final_test_set_goals.pickle', 'rb') as handle:
        agent_pos_goals, cart_x_goals, breads_goals = pickle.load(handle)
else:
    print("TEST SET NOT FOUND: run generate_random_goals script first")

first_floor = scale_vector([4.],np.array(b.get_bounds(['agent_y'])))[0]
snd_floor = scale_vector([6.],np.array(b.get_bounds(['agent_y'])))[0]
snd_floor_limit = scale_vector([440.7],np.array(b.get_bounds(['agent_z'])))[0]
# clean agent_pos_goals
for g in agent_pos_goals:
    print(g)
    if g[2] < snd_floor_limit:
        g[1] = first_floor
    else:
        g[1] = snd_floor
    print('after: %s' % g)
print("agent goals cleaned up")

# final competence test on goals choosen by engineer
print(np.array(full_outcome_bounds)[[0,1,2]])
print(np.array(full_outcome_bounds)[[3]])
print(np.array(full_outcome_bounds)[[4]])

agent_pos_goals = scale_vector(np.array([[294.5,6.,443.5],[288.5,6.,443.5],[293.5,4.,436.3]]),
                               np.array(full_outcome_bounds)[[0,1,2]])
cart_x_goals = scale_vector(np.array([[296.5],[286.5],[294.5]]),
                        np.array(full_outcome_bounds)[[3]])
breads_goals = scale_vector(np.array([[0.],[1.],[2.],[3.],[4.],[5.]]),
                        np.array(full_outcome_bounds)[[4]])

e_agent, e_cart, e_breads, cart_touched = eval(agent_pos_goals[-3:], cart_x_goals, breads_goals)
b_k['final_eval_errors'] = [e_agent, e_cart, e_breads]
b_k['final_eval_cart_touched'] = cart_touched
'''
print("saving gep")
save_gep(gep, max_iterations, b_k, savefile_name, book_keeping_file_name)