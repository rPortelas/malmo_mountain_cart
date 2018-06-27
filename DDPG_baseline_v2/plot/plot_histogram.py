import os
import matplotlib.pyplot as plt
import numpy as np

directory = "./first/"

def plot_histo():
    n_runs = 1200

    n_steps_success = np.zeros([n_runs])
    cpt = 0
    for name in os.listdir(directory):
        path = directory + name + "/"
        for file in os.listdir(path):
            nb_steps = np.loadtxt(file)
            cpt += 1
            print(nb_steps)
            n_steps_success[cpt] = nb_steps[0]

    print('mean number of steps before reaching a reward: ', n_steps_success.mean())
    fig = plt.figure()
    plt.hist(n_steps_success, )
    plt.xlabel('number of steps before first reward')
    plt.title('Histogram of the number of steps before reaching the first reward (' + str(n_runs) + ' runs)')

def plot_histo2():
    file = "collec.steps"
    nb_steps_tab = np.loadtxt(file)
    print('mean number of steps before reaching a reward: ', nb_steps_tab.mean())
    fig = plt.figure()
    plt.hist(nb_steps_tab,bins=50 )
    plt.xlabel('number of steps before first reward')
    plt.title('Histogram of the number of steps before reaching the first reward (' + str(len(nb_steps_tab)) + ' runs)')
    plt.savefig("histo_DDPG.png")

plot_histo2()
