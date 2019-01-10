import torch
import numpy as np


def uniform_init(layers, params):

    scale = params['scale']
    dims = params['dims']

    nb_layers = len(layers)
    if nb_layers > 0:
        nb_weights = dims['s'] * layers[0]
        for i in range(1, nb_layers):
            nb_weights += layers[i - 1] * layers[i]
        nb_weights += layers[-1] * dims['a']
    else:
        nb_weights = dims['s'] * dims['a']

    weights = np.random.uniform(-scale, scale, nb_weights)

    nb_biases = sum(layers) + dims['a']
    biases = np.zeros([nb_biases])

    return weights, biases


def gloriot_uniform(layers, params):
    # glorot uniform initialization
    # does not work
    dims = params['dims']

    nb_layers = len(layers)

    weights = np.array([]).reshape([0])
    biases = np.array([]).reshape([0])
    for i in range(nb_layers + 1):
        if i == 0:
            n_in = dims['s']
        else:
            n_in = layers[i - 1]
        if i == nb_layers:
            n_out = dims['a']
        else:
            n_out = layers[i]

        size_layer = n_in * n_out
        layer_weights = np.random.uniform(- np.sqrt(6 / (n_in + n_out)), np.sqrt(6 / (n_in + n_out)), size_layer)
        weights = np.concatenate([weights, layer_weights], axis=0)
        biases = np.concatenate([biases, np.zeros([n_out])])

    return weights, biases

def he_uniform(layers, params):
    # he uniform initialization
    # works for tanh and relu !
    dims = params['dims']

    nb_layers = len(layers)

    weights = np.array([]).reshape([0])
    biases = np.array([]).reshape([0])
    for i in range(nb_layers + 1):
        if i == 0:
            n_in = dims['s']
        else:
            n_in = layers[i - 1]
        if i == nb_layers:
            n_out = dims['a']
        else:
            n_out = layers[i]

        size_layer = n_in * n_out
        layer_weights = np.random.uniform(- np.sqrt(6 / n_in), np.sqrt(6 / n_in), size_layer)
        weights = np.concatenate([weights, layer_weights], axis=0)
        biases = np.concatenate([biases, np.zeros([n_out])])

    return weights, biases

def lecun_normal_uniform(layers, params):
    # lecun normal initialization for the first layer
    # lecun uniform initialization for the subsequent layers

    dims = params['dims']

    nb_layers = len(layers)

    weights = np.array([]).reshape([0])
    biases = np.array([]).reshape([0])
    for i in range(nb_layers + 1):
        if i == 0:
            n_in = dims['s']
            scaling = 1
        else:
            n_in = layers[i - 1]
            scaling = 3
        if i == nb_layers:
            n_out = dims['a']
        else:
            n_out = layers[i]

        size_layer = n_in * n_out
        layer_weights = np.random.uniform(- np.sqrt(scaling / n_in), np.sqrt(scaling / n_in), size_layer)
        weights = np.concatenate([weights, layer_weights], axis=0)
        biases = np.concatenate([biases, np.zeros([n_out])])

    return weights, biases

def lecun_uniform(layers, params):
    # lecun uniform initialization

    dims = params['dims']

    nb_layers = len(layers)

    weights = np.array([]).reshape([0])
    biases = np.array([]).reshape([0])
    for i in range(nb_layers + 1):
        if i == 0:
            n_in = dims['s']
        else:
            n_in = layers[i - 1]
        if i == nb_layers:
            n_out = dims['a']
        else:
            n_out = layers[i]

        size_layer = n_in * n_out
        layer_weights = np.random.uniform(- np.sqrt(3 / n_in), np.sqrt(3 / n_in), size_layer)
        weights = np.concatenate([weights, layer_weights], axis=0)
        biases = np.concatenate([biases, np.zeros([n_out])])

    return weights, biases

def he_normal_uniform(layers, params):
    # he normal initialization for the first layer
    # he uniform initialization for the subsequent layer
    # He et al., 2015, works for tnah and relu !

    dims = params['dims']

    nb_layers = len(layers)

    weights = np.array([]).reshape([0])
    biases = np.array([]).reshape([0])

    for i in range(nb_layers + 1):
        if i == 0:
            n_in = dims['s']
            scaling = 2
        else:
            n_in = layers[i - 1]
            scaling = 6
        if i == nb_layers:
            n_out = dims['a']
        else:
            n_out = layers[i]
        size_layer = n_in * n_out
        layer_weights = np.random.uniform(- np.sqrt(scaling / (n_in)), np.sqrt(scaling / (n_in)), size_layer)
        weights = np.concatenate([weights, layer_weights], axis=0)
        biases = np.concatenate([biases, np.zeros([n_out])])

    return weights, biases

def gloriot_normal_uniform(layers, params):
    # glorot normal initialization for the first layer
    # glorot uniform initialization for subsequent layer
    # does not work..
    dims = params['dims']

    nb_layers = len(layers)

    weights = np.array([]).reshape([0])
    biases = np.array([]).reshape([0])

    for i in range(nb_layers + 1):
        if i == 0:
            n_in = dims['s']
            scaling = 2
        else:
            n_in = layers[i - 1]
            scaling = 6
        if i == nb_layers:
            n_out = dims['a']
        else:
            n_out = layers[i]
        size_layer = n_in * n_out
        layer_weights = np.random.uniform(- np.sqrt(scaling / (n_in + n_out)), np.sqrt(scaling / (n_in + n_out)), size_layer)
        weights = np.concatenate([weights, layer_weights], axis=0)
        biases = np.concatenate([biases, np.zeros([n_out])])

    return weights, biases