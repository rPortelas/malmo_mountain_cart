import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np


class PolicyNN(nn.Module):

    def __init__(self, params):
        super(PolicyNN, self).__init__()

        self.layers = params['layers']
        self.activation = params['activation_function']
        self.max_action = params['max_a']
        self.dims = params['dims']
        self.add_bias = params['bias']
        self.nb_layers = len(self.layers)

        # create the layers
        if self.nb_layers > 0:
            self.l1 = nn.Linear(self.dims['s'], self.layers[0])
            for i in range(1, self.nb_layers):
                self.__setattr__("l{}".format(i + 1), nn.Linear(self.layers[i - 1], self.layers[i]))
            self.__setattr__("l{}".format(self.nb_layers + 1), nn.Linear(self.layers[-1], self.dims['a']))
        else:
            self.l1 = nn.Linear(self.dims['s'], self.dims['a'])

        for l in self._modules:
            self._modules[l].bias.data.zero_()

        # list number of weights and biases per layer
        self.nb_biases = [self._modules[l].bias.data.size()[0] for l in self._modules]
        self.nb_weights = [np.prod(self._modules[l].weight.data.size()) for l in self._modules]

        self.nb_total_biases = sum(self.nb_biases)
        print('nb_biases: {}'.format(self.nb_total_biases))
        self.weights = None
        self.biases = None

    def get_action(self, state):
        x = torch.FloatTensor(state)
        if self.nb_layers > 0:
            for i in range(self.nb_layers):
                l_current = list(self._modules.items())[i][1]
                if self.activation == 'relu':
                    x = func.relu(l_current(x))
                elif self.activation == 'tanh':
                    x = torch.tanh(l_current(x))
        l_last = list(self._modules.items())[-1][1]
        x = self.max_action * torch.tanh(l_last(x))
        return x.detach().numpy()

    def set_parameters(self, parameters):
        # In: list of pytorch parameters
        # set parameters to each layer of the model
        weights = parameters[0:-self.nb_total_biases]
        biases = parameters[-self.nb_total_biases:]
        ##print(len(weights))
        #print(len(biases))
        self.weights = weights
        self.biases = biases
        ind_weights = 0
        ind_biases = 0
        for i_m, m in enumerate(self._modules):
            self._modules[m].weight.data = torch.FloatTensor(weights[ind_weights: ind_weights + self.nb_weights[i_m]].reshape(self._modules[m].weight.data.size()))
            self._modules[m].bias.data = torch.FloatTensor(biases[ind_biases: ind_biases + self.nb_biases[i_m]])
            ind_weights += self.nb_weights[i_m]
            ind_biases += self.nb_biases[i_m]

        assert ind_weights == sum(self.nb_weights)
        assert ind_biases == sum(self.nb_biases)

    def initialize_weights(self):
        self.weights, self.biases = self.init_function(self.layers, self.params_init)
        self.set_parameters(self.weights, self.biases)



    def get_parameters(self):
        return self.weights.copy(), self.biases.copy()

    def logs(self, prefix=''):
        logs = []
        # logs += [('stats_o/mean', np.mean(self.o_stats.mean))]
        # logs += [('stats_o/std', np.mean(self.o_stats.std))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs
