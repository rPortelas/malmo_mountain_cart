import numpy as np

class Sequential_NN(object):
    def __init__(self, max_steps, seq_size, in_size, in_bounds, out_size, hidden_size, out_activation="tanh"):
        self.max_steps = max_steps
        self.steps_per_nn = int(max_steps / seq_size)
        print('Initiating sequential NN, with {} steps per nn'.format(self.steps_per_nn))
        self.NN = Simple_NN(in_size, in_bounds, out_size, hidden_size, out_activation)
        self.w_list = None
        self.step_counter = -1

    def set_weights(self, w_list):
        self.w_list = w_list
        self.step_counter = 0

    def forward(self, state):
        assert(self.step_counter < self.max_steps)
        w_idx = self.step_counter // self.steps_per_nn
        #print(w_idx)
        self.step_counter += 1
        return self.NN.forward(state, self.w_list[w_idx])




class Simple_NN(object):
    # simple 1 hidden layer neural network with relu then tanh activation
    def __init__(self, in_size, in_bounds, out_size, hidden_size, out_activation="tanh"):
        assert(in_size == len(in_bounds))
        self.in_size = in_size
        self.in_bounds = np.array(in_bounds)
        self.out_size = out_size
        self.hidden_size = hidden_size
        if self.in_size == 9:
            self.tmp_controller = 0.24 #0.25
        elif self.in_size == 10:
            self.tmp_controller = 0.21
        elif self.in_size == 12:
            self.tmp_controller = 0.18
        else:
            raise NotImplementedError

        self.nb_w1_weights = self.hidden_size*self.in_size
        self.nb_w2_weights = self.hidden_size*self.out_size

    def forward(self, x, weights, scale=True): #parameters are assumed to be numpy arrays
        assert(len(weights) == (self.nb_w1_weights + self.nb_w2_weights))

        #scale input to [-1,1]
        if scale:
            mins_maxs_diff =  np.diff(self.in_bounds).squeeze()
            mins = self.in_bounds[:, 0]
            x = (x - mins) * 2 / mins_maxs_diff - 1

        # create weight arrays
        w1 = weights[0:self.nb_w1_weights].reshape(self.in_size,self.hidden_size)
        w2 = weights[self.nb_w1_weights:].reshape(self.hidden_size,self.out_size)

        # forward propagate input in network and compute output activation
        h = np.dot(x,w1)
        h[h<0] = 0 # relu activation
        #print h
        #print h
        out = np.dot(h,w2)
        #print out
        return np.tanh(out * self.tmp_controller)[0,:]



