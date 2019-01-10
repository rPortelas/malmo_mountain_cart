import time
import numpy as np
import matplotlib.pyplot as plt
from neural_network import PolicyNN
from initialization_functions import he_uniform

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# init neural network policy
state_size = 9
action_set_size = 3
layers = [64]
params = {'layers': layers, 'activation_function':'relu', 'max_a':1.,
          'dims':{'s':state_size,'a':action_set_size},'bias':True}
model = PolicyNN(params)

# test output distribution when generating weights with he_uniform and random input values
nb_iterations = 100000
outputs = np.zeros((nb_iterations, action_set_size))
#temperature_param = [0.14] # 0.17 (or 0.12 when input=21) works great !
for k in range(nb_iterations//100):
    weights, biases = he_uniform(layers, params)
    for i in range(nb_iterations//1000):
        # random input
        x = np.random.random(state_size) * 2 - 1
        model.set_parameters(np.concatenate((weights,biases)))
        out = model.get_action(x.reshape(1,-1))
        end = time.time()
        #print out
        #print out.shape
        outputs[(k*(nb_iterations//1000))+i,:] = out
        #print out

plt.figure(1)
plt.hist(outputs[:,0])
plt.figure(2)
plt.hist(outputs[:,1])
plt.figure(3)
plt.hist(outputs[:, 2])
plt.show()

