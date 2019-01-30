import numpy as np
from collections import OrderedDict
from utils.initialization_functions import he_uniform

# scale numpy 1d array to [-1:1] given its bounds
# bounds must be of the form [[min1,max1],[min2,max2],...]
def scale_vector(values, bounds):
    mins_maxs_diff =  np.diff(bounds).squeeze()
    mins = bounds[:, 0]
    return (((values - mins) * 2) / mins_maxs_diff) - 1

def unscale_vector(scaled_values, bounds=[[-1,1]]):
    mins_maxs_diff =  np.diff(bounds).squeeze()
    mins = bounds[:, 0]
    return (((scaled_values + 1) * mins_maxs_diff) / 2) + mins

def proportional_choice(v, eps=0.):
    if np.sum(v) == 0 or np.random.rand() < eps:
        return np.random.randint(np.size(v))
    else:
        probas = np.array(v) / np.sum(v)
        return np.where(np.random.multinomial(1, probas) == 1)[0][0]

# def get_random_policy(layers, init_function_params):
#     rnd_weights, rnd_biases = he_uniform(layers, init_function_params)
#     current_policy = np.concatenate((rnd_weights, rnd_biases))
#     return current_policy

def get_random_nn(layers, init_function_params):
    rnd_weights, rnd_biases = he_uniform(layers, init_function_params)
    return np.concatenate((rnd_weights, rnd_biases))


def get_random_policy(layers, init_function_params):
    policy = []
    for i in range(init_function_params['size_sequential_nn']):
        policy.append(get_random_nn(layers, init_function_params).copy())
    return policy



class Bounds(object):
    def __init__(self):
        # define state variables' bounds
        # they will be used for our policy input and our outcome space
        self.bounds = OrderedDict()

    def add(self, name, bounds):
        self.bounds[name] = bounds

    def get_bounds(self, name_list):
        return [self.bounds[n] for n in name_list]


class Distractors(object):
    def __init__(self, nb_distractors=2, noise=0.1):
        self.nb_ds = nb_distractors
        self.noise = noise
        self.ds = None

    def reset(self):
        self.ds = np.random.rand(self.nb_ds)

    def step(self):
        self.ds += np.random.normal(0, self.noise,len(self.ds))
        self.ds = np.clip(self.ds,-1.,1.)

    def get(self):
        return self.ds.copy()




