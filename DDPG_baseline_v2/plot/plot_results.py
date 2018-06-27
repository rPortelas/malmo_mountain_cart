import json
from perfcollector_dataframe import PerfCollectorData
import os
import matplotlib.pyplot as plt

directory = "../hc_results/" #"../hop_results/" #"../cmc_geppg_nonoise_nofrozen/" #"./perf_ofp/" #"./experiments_tau/" #"./results/perf_ofp_GEP/"  #

def get_perf_values(filename):
    with open(filename, 'r') as json_data:

        lines = json_data.readlines()
        eval_rewards = []
        for line in lines:

            episode_data = json.loads(line)
            if 'New Test reward' in episode_data:
                step = episode_data['New Training steps']
                perf = episode_data['New Test reward']
                tmp = [step, perf]
                eval_rewards.append(tmp)
    return eval_rewards

def plot_all():
    cpt = 0
    collector = PerfCollectorData("../img/")
    for delta in os.listdir(directory):
        collector.init(delta)
        perf_values = {}
        experiment_path = directory + delta + "/"
        for file in os.listdir(experiment_path):
            filename = experiment_path + file + "/log_episodes/progress.json"
            perf_values = get_perf_values(filename)
            cpt += 1
            collector.add(delta, perf_values)
        collector.plot(delta)
    print(cpt, " files found")
    collector.stats()
    collector.plot_all()

def plot_all_enum():
    perf_values = []
    for name in os.listdir(directory):
        experiment_path = directory + name + "/"
        for file in os.listdir(experiment_path):
            filename = experiment_path + file + "/log_episodes/progress.json"
            perf_values.append(get_perf_values(filename))
    plot_local(perf_values)

def plot_local(perf_values):
    plt.figure(1, figsize=(20,13))
    plt.xlabel("time steps")
    plt.ylabel("performance")
    plt.title("Performance")

    for values in perf_values:
        if len(values)>1:
            x = []
            y = []
            for i in range(len(values)):
                x.append(values[i][0])
                y.append(values[i][1])
            plt.plot(x,y, label="label")
    #plt.legend()
    #plt.show()
    plt.savefig("../img/perf.png", bbox_inches='tight')

plot_all_enum()
