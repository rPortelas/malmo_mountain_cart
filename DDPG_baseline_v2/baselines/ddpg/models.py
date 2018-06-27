import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import torch

class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]



class Actor(Model):
    def __init__(self, nb_actions, nb_observations, name='actor', layer_norm=True, initial_weights=None):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

        self._layer_sizes = [nb_observations] + [64,64] + [nb_actions]
        self._initial_weights = initial_weights
        self._bool = False

        if self._initial_weights is not None:
            self._initialize = True
        else:
            self._initialize = False

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs

            if self._initialize:
                if not self._bool:
                    self._bool = True
                    self._weights_tf = []
                    self._biases_tf = []
                    for i in range(len(self._layer_sizes) - 1):
                        shape_weights = (self._layer_sizes[i], self._layer_sizes[i + 1])
                        self._weights_tf.append(
                            tf.get_variable('weights' + str(i), initializer=tf.zeros_initializer(), shape=shape_weights))
                        shape_biases = (self._layer_sizes[i + 1],)
                        self._biases_tf.append(
                            tf.get_variable('biases' + str(i), initializer=tf.zeros_initializer(), shape=shape_biases))

                x = tf.add(tf.matmul(x, self._weights_tf[0]), self._biases_tf[0])
                for i in range(1,len(self._layer_sizes)-1):
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    x = tf.nn.relu(x)
                    x = tf.add(tf.matmul(x, self._weights_tf[i]), self._biases_tf[i])
                x = tf.nn.tanh(x*0.03)
            else:
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
                x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = tf.nn.tanh(x)
        return x

    def set_initial_weights(self, sess):

        if self._initialize:
            assert self._initial_weights.ndim == 1
            print('loading initial weights')
            index = 0
            # define assignment operations to fill weight and bias tf variables using controller_params values
            for i in range(len(self._layer_sizes) - 1):
                ind_weights = np.arange(index, index + self._layer_sizes[i] * self._layer_sizes[i + 1])
                index = index + (self._layer_sizes[i]) * self._layer_sizes[i + 1]
                # if i == len(self._layer_sizes) - 2:
                #     self._initial_weights[ind_weights] *= 0.03 # equivalent of the 0.03 slope of tanh in gep
                weights = self._initial_weights[ind_weights].reshape([self._layer_sizes[i], self._layer_sizes[i + 1]])
                assignment = self._weights_tf[i].assign(weights)
                sess.run(assignment)



class Critic(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
