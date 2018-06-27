import pickle
import numpy as np
import matplotlib.pyplot as plt

folder_location = './CMC_buffer_tanh'
n_runs = 50

n_steps_success = np.zeros([n_runs])
for i in range(n_runs):
    loading_path = folder_location+'/simu_CMC3_'+str(i+1)+'_buffer'
    file = open(loading_path, 'rb')
    buffer = pickle.load(file)
    file.close()

    # compute number of steps before reaching the first reward
    # 1000 steps times the number of unsuccessful episodes + number of steps of the successful episode (last one)
    n_unsuccessful = len(buffer)-1
    n_steps_success[i] = 1000*n_unsuccessful+len(buffer[-1])


print('mean number of steps before reaching a reward: ', n_steps_success.mean())
fig = plt.figure()
plt.hist(n_steps_success,)
plt.xlabel('number of steps before first reward')
plt.title('Histogram of the number of steps before reaching the first reward ('+str(n_runs)+' runs)')

