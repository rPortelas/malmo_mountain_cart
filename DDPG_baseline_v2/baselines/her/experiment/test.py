import sys
sys.path.append('/home/remy/armball-her/')
import gym2
import numpy as np
env = gym2.make('MalmoMountainCart-v0')

for j in range(10):
    out = env.reset()
    #out  = env.reset()
    obs = out['observation']
    goal = out['desired_goal']
    for i in range(70):
       #print(i)
       action = np.random.uniform(-1,1,2)
       out,_,_,_ = env.step(action)
       obs= out['observation']
       goal = out['desired_goal']
       achieved_goal = out['achieved_goal']