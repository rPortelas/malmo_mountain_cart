from __future__ import division
import numpy as np
from collections import OrderedDict

# scale numpy 1d array to [-1:1] given its bounds
# bounds must be of the form [[min1,max1],[min2,max2],...]
def scale_vector(values, bounds):
    mins_maxs_diff =  np.diff(bounds).squeeze()
    mins = bounds[:, 0]
    return (((values - mins) * 2) / mins_maxs_diff) - 1

def unscale_vector(scaled_values, bounds=[-1,1]):
    mins_maxs_diff =  np.diff(bounds).squeeze()
    mins = bounds[:, 0]
    return (((scaled_values + 1) * mins_maxs_diff) / 2) + mins

def proportional_choice(v, eps=0.):
    if np.sum(v) == 0 or np.random.rand() < eps:
        return np.random.randint(np.size(v))
    else:
        probas = np.array(v) / np.sum(v)
        return np.where(np.random.multinomial(1, probas) == 1)[0][0]

class Bounds(object):
    def __init__(self):
        # define state variables' bounds
        # they will be used for our policy input and our outcome space
        self.bounds = OrderedDict()

    def add(self, name, bounds):
        self.bounds[name] = bounds

    def get_bounds(self, name_list):
        return [self.bounds[n] for n in name_list]