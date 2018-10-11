import os
import os.path
import sys
import time
import json
import pickle
import time
import numpy as np
from gep import GEP
from nn_policy import Simple_NN
import matplotlib.pyplot as plt
from plot_utils import plot_agent_pos_exploration, plot_agent_cart_exploration, plot_agent_bread_exploration
import collections


# init neural network policy
state_bounds = np.array([[-1.,1.]]*9)
hidden_layer_size = 64
state_size = 9
action_set_size = 2
model = Simple_NN(state_size, state_bounds, action_set_size , hidden_layer_size)
policy_nb_dims = model.nb_w1_weights + model.nb_w2_weights

# test output distribution when generating random weight and input values
nb_iterations = 100000
outputs = np.zeros((nb_iterations, action_set_size))

temperature_param = [0.24,0.28] # 0.17 (or 0.12 when input=21) works great !
for tmp in temperature_param:
    print("using tanh temperature of %s" % temperature_param)
    model.tmp_controller = tmp
    print(model.tmp_controller)
    # random weights
    for k in range(nb_iterations//100):
        weights = np.random.random(policy_nb_dims) * 2 - 1
        for i in range(nb_iterations//1000):
            # random input
            x = np.random.random(state_size) * 2 - 1
            for j,v in enumerate(x[8:].tolist()):
                if v < 0:
                    x[8+j] = -1.
                else:
                    x[8+j] = 1.

            out = model.forward(x.reshape(1,-1),weights,scale=False)
            #print out
            #print out.shape
            outputs[(k*(nb_iterations//1000))+i,:] = out
            #print out

    #print outputs[:][0].shape
    plt.figure(1)
    plt.hist(outputs[:,0])
    plt.figure(2)
    plt.hist(outputs[:,1])
    plt.show(block=False)
    time.sleep(2.)
    plt.close(1)
    plt.close(2)

