import gym2
import numpy as np
env = gym2.make('MalmoMountainCart-v0')


out = env.reset()
obs = out['observation']
goal = out['desired_goal']
for i in range(70):
   action = np.random.uniform(-1,1,2)
   out = env.step(action)
   obs = out[0]['observation']
   goal = out[0]['desired_goal']
   achieved_goal = out[0]['achieved_goal']