from __future__ import division
import gym2
from gym2 import spaces
from gym2.utils import seeding
import numpy as np
import MalmoPython
import json
import time
import random
import sys
import getpass
import os
sys.path.append('/home/rportelas/malmo_mountain_cart/')
from utils.gep_utils import Bounds, unscale_vector
# place bread at given positions
def draw_bread(bread_positions):
    xml_string = ""
    for x,y,z in bread_positions:
        xml_string += '<DrawItem x="%s" y="%s" z="%s" type="bread"/>' % (int(x),int(y),int(z))
        xml_string += '\n'
    return xml_string

# erase previous items in defined bread positions
def clean_bread(bread_positions):
    xml_string = ""
    for x,y,z in bread_positions:
        xml_string += '<DrawBlock x="%s" y="%s" z="%s" type="air"/>' % (int(x),int(y),int(z))
    return xml_string

def get_MMC_environment(bread_positions, tick_lengths, skip_step, desired_mission_time, minecraft_dir, mission_start_sleep=0.5):
    total_allowed_actions = int((20/(skip_step+1)) * desired_mission_time) # dependent of skip_step, works if =1
    # if big overclocking, set display refresh rate to 1
    mod_setting = '' if tick_lengths >= 25 else "<PrioritiseOffscreenRendering>true</PrioritiseOffscreenRendering>"
    #print("tick: {}, desiredmtime: {}".format(tick_lengths, desired_mission_time))
    mission_time_limit = str((50 * 20 * desired_mission_time + (50/tick_lengths)*2*(1000*mission_start_sleep)))
    #print(mission_time_limit)
    missionXML = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

              <About>
                <Summary>Goal Exploration Process, in Malmo !</Summary>
              </About>

              <ModSettings>
              <MsPerTick>''' + str(tick_lengths) + '''</MsPerTick>
              ''' + mod_setting + '''
              </ModSettings>

              <ServerSection>
                <ServerInitialConditions>
                  <Weather>clear</Weather>
                  <Time>
                  <StartTime>12000</StartTime>
                  <AllowPassageOfTime>false</AllowPassageOfTime>
                  </Time>
                </ServerInitialConditions>
                <ServerHandlers>
                  <FileWorldGenerator src="/home/'''+getpass.getuser()+minecraft_dir+'''Minecraft/run/saves/flowers_v4"/>
                  <DrawingDecorator>
                    <DrawLine x1="288" y1="6" z1="443" x2="294" y2="6" z2="443" type="air"/>
                    <DrawLine x1="287" y1="7" z1="443" x2="295" y2="7" z2="443" type="air"/>
                    <DrawLine x1="286" y1="8" z1="443" x2="296" y2="8" z2="443" type="air"/>

                    ''' + clean_bread(bread_positions) + '''

                    <DrawBlock x="287" y="7" z="443" type="rail"/>
                    <DrawBlock x="286" y="8" z="443" type="rail"/>
                    <DrawBlock x="295" y="7" z="443" type="rail"/>
                    <DrawBlock x="296" y="8" z="443" type="rail"/>
                    <DrawLine x1="288" y1="6" z1="443" x2="294" y2="6" z2="443" type="rail"/>
                    <DrawEntity x="291.5" y="6" z="443" type="MinecartRideable"/>

                    ''' + draw_bread(bread_positions) + '''

                  </DrawingDecorator>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>FlowersBot</Name>
                <AgentStart>
                  <Placement x="293.5" y="4" z="433.5" yaw="0"/>
                  <Inventory></Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromFullInventory flat="false"/>
                  <AbsoluteMovementCommands/>
                  <ObservationFromNearbyEntities>
                    <Range name="entities" xrange="40" yrange="40" zrange="40"/>
                  </ObservationFromNearbyEntities>
                  <ObservationFromFullStats/>
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                  <MissionQuitCommands/>
                  <AgentQuitFromReachingCommandQuota total="''' + str((2 * total_allowed_actions)) + '''"/>
                  <AgentQuitFromTimeUp timeLimitMs="''' + mission_time_limit + '''"/>
                  <VideoProducer>
                    <Width>400</Width>
                    <Height>300</Height>
                  </VideoProducer>
                </AgentHandlers>

              </AgentSection>
            </Mission>'''
    return missionXML

class MalmoMountainCart(gym2.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
                    #  <ServerQuitFromTimeUp timeLimitMs="''' + mission_time_limit + '''"/>
    def __init__(self, port=10000, tick_lengths=10, skip_step=1, desired_mission_time=7, sparse=False, reward_mixing=20):
        print('Making new MMC instance')
        self.skip_step = skip_step
        self.tick_lengths = tick_lengths
        self.total_allowed_actions = 10 * desired_mission_time #dependent of skip_step, works if =1
        self._sparse = sparse
        self._reward_mixing = reward_mixing

        # define bread positions in MMC arena
        self.mission_start_sleep = 0.1
        self.bread_positions = [[293.5,4,436.5],[289.5,4,437.5],[289.5,4,440.5],[291.5,6,442.5],[294.5,6,443.5]]

        #load Minecraft version name
        self.minecraft_dir = '/' + os.environ['MALMO_DIR'] + '/'
        print(self.minecraft_dir)
        self.mission_xml = get_MMC_environment(self.bread_positions, 
                                               tick_lengths,
                                               skip_step,
                                               desired_mission_time,
                                               self.minecraft_dir,
                                               mission_start_sleep=self.mission_start_sleep)
        # Create default Malmo objects:
        self.agent_host = MalmoPython.AgentHost()
        self.my_mission = MalmoPython.MissionSpec(self.mission_xml, True)
        self.my_mission_record = MalmoPython.MissionRecordSpec()

        self.client_pool = MalmoPython.ClientPool()

        #print("Attempt to communicate with Minecraft through port %s" % port)
        # enable the use of up to 21 parallel malmo mountain carts
        for i in range(20):
            self.client_pool.add(MalmoPython.ClientInfo( "127.0.0.1", port+i))

        n_act = 2
        n_obs = 9

        self.action_space = spaces.Box(low=-np.ones(n_act), high=np.ones(n_act))
        self.observation_space = spaces.Box(low=-np.repeat([np.array([np.inf])], n_obs), high=np.repeat([np.array([np.inf])], n_obs))

        # init goal sampling
        self.state_names = ['agent_x','agent_y','agent_z','cart_x'] + ['bread_'+str(i) for i in range(5)]
        self.b = Bounds()
        self.b.add('agent_x',[288.3,294.7])
        self.b.add('agent_y',[4,6])
        self.b.add('agent_z',[433.3,443.7])
        self.b.add('cart_x',[285,297])
        for i in range(5):
            self.b.add('bread_'+str(i),[0,1])

        self.current_step = 0

        self.seed()
        #self.reset()

    # call this method to change default parameters
    def my_init(self, port=10000, tick_lengths=10, skip_step=1, desired_mission_time=7):
        self.skip_step = skip_step
        self.tick_lengths = tick_lengths
        self.total_allowed_actions = int((20/(skip_step+1)) * desired_mission_time)
        self.mission_xml = get_MMC_environment(self.bread_positions, 
                                               tick_lengths,
                                               skip_step,
                                               desired_mission_time,
                                               self.minecraft_dir,
                                               mission_start_sleep=self.mission_start_sleep)
        # Create default Malmo objects:
        self.agent_host = MalmoPython.AgentHost()
        self.my_mission = MalmoPython.MissionSpec(self.mission_xml, True)
        self.my_mission_record = MalmoPython.MissionRecordSpec()

        self.client_pool = MalmoPython.ClientPool()

        print("Attempt to communicate with Minecraft through port %s" % port)
        self.client_pool.add(MalmoPython.ClientInfo( "127.0.0.1", port))


    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return seed

    def sample_goal(self):
        goal = np.random.random(9) * 2 - 1
        env_scaled_goal = unscale_vector(goal, np.array(self.b.get_bounds(self.state_names)))
        env_scaled_goal[4:] = np.array([0 if b < 0.5 else 1 for b in env_scaled_goal[4:]])
        #print("goal sampled: {}".format(env_scaled_goal))
        return env_scaled_goal
        

    def get_world_state(self, first_state=False):
        # wait till we have got at least one observation or mission has ended
        while True:
            time.sleep(0.001)  # wait for 1ms to not consume entire CPU
            world_state = self.agent_host.peekWorldState()
            #print(world_state.number_of_observations_since_last_state)
            if world_state.number_of_observations_since_last_state > (self.skip_step+1):
                if not first_state:
                    print("DAMMIT, WE LOST %s OBSERVATION" % (world_state.number_of_observations_since_last_state - self.skip_step))
                    print(self.current_step)
            if world_state.number_of_observations_since_last_state > self.skip_step or not world_state.is_mission_running:
                break
        return self.agent_host.getWorldState()

    def reset(self, goal=None):
        #print('Resetting mission')
        # world_state = self.get_world_state()
        # print(world_state.has_mission_begun)
        # if world_state.has_mission_begun: # if true another mission is still running
        #     print(self.current_step)
        #     print('*** Aborting previous mission ***')
        #     self.agent_host.sendCommand("quit")
        world_state = self.agent_host.peekWorldState()
        #print("resetting, previous mission running ?: {}".format(world_state.is_mission_running))
        if world_state.is_mission_running:
            self.agent_host.sendCommand("quit")
            while world_state.is_mission_running:
                print('waiting mission abortion...')
                time.sleep(0.01)
                world_state = self.agent_host.peekWorldState()
            print('aborted')

        # Attempt to start a mission:
        max_retries = 5
        sleep_time = [0.01, 0.1, 2., 2., 5.]
        for retry in range(max_retries):
            try:
                self.agent_host.startMission(self.my_mission,
                                             self.client_pool,
                                             self.my_mission_record,
                                             0, "answer is 42")
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(sleep_time[retry])

        # Loop until mission starts:
        world_state = self.agent_host.peekWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.001)
            world_state = self.agent_host.peekWorldState()
        # dirty solution to wait for client to be stable
        time.sleep(self.mission_start_sleep)
        world_state = self.get_world_state(first_state=True)
        obvsText = world_state.observations[-1].text
        observation = json.loads(obvsText)  # observation comes in as a JSON string...
        self.current_step = 0

        self.desired_goal = self.sample_goal()
        state = self.get_state(observation)
        reward = self.compute_reward(state, self.desired_goal)

        obs = dict(observation=state,
                   achieved_goal=state,
                   desired_goal=self.desired_goal)
        done = False
        #return obs, reward, done, {}
        #print(obs)
        return obs

    def get_state(self, obs):
        breads = np.ones(len(self.bread_positions))
        for e in obs['entities']:
            if e['name'] == 'MinecartRideable':
                cart_x, cart_y, cart_z = e['x'], e['y'], e['z']
                cart_vx, cart_vy, cart_vz = e['motionX'], e['motionY'], e['motionZ']
            if e['name'] == 'FlowersBot':
                agent_x, agent_y, agent_z, agent_yaw = e['x'], e['y'], e['z'], (e['yaw'] % 360)
                agent_vx, agent_vy, agent_vz = e['motionX'], e['motionY'], e['motionZ']
            if e['name'] == 'bread':
                pos = [e['x'],e['y'],e['z']]
                bread_idx = self.bread_positions.index(pos) # current bread must be one of the positioned bread
                breads[bread_idx] = 0 #if bread is in arena it's not in our agent's pocket, so 0
        return np.array([agent_x, agent_y, agent_z, cart_x] + breads.tolist())


    def step(self, actions):
        # format actions for environment
        actions = ["move " + str(actions[0]), "strafe " + str(actions[1])]
        self.current_step += 1
        #print(self.current_step)
        done = False
        # print self.current_step
        # take the action only if mission is still running
        world_state = self.agent_host.peekWorldState()
        #print(self.current_step)
        if world_state.is_mission_running:
            # take action
            if self.current_step == self.total_allowed_actions:  # end of episode
                # last cmd, must teleport to avoid weird glitch with minecart environment
                self.agent_host.sendCommand("tp 293 7 433.5")
                # send final dummy action
                self.agent_host.sendCommand("move 0")
                done = False
                world_state = self.agent_host.peekWorldState()
                while world_state.is_mission_running:
                    #print('waiting for end of mission')
                    time.sleep(0.01)
                    world_state = self.agent_host.peekWorldState()
                done = True
            else:
                for a in actions:
                    self.agent_host.sendCommand(a)

                # wait for the new state
                if self.current_step == 1: #first time we acted, discard missed observation warning
                    world_state = self.get_world_state(first_state=True)
                else:
                    world_state = self.get_world_state()

                # log errors and control messages
                for error in world_state.errors:
                    print('error.text')
        if not done:
            obvsText = world_state.observations[-1].text
            observation = json.loads(obvsText)  # observation comes in as a JSON string...
            state = self.get_state(observation)
            self.previous_state = state
        else:
            state = self.previous_state  # last state is state n-1
            # detect terminal state
            # print not world_state.is_mission_running

        reward = self.compute_reward(state, self.desired_goal)
        info = {'is_success': bool(reward == 1)}

        obs = dict(observation=state,
                   achieved_goal=state,
                   desired_goal=self.desired_goal)
        #print(obs)
        #print('reward: {}'.format(reward))
        return obs, reward, done, info
        # for ddpg
        # return state, reward, done, {}

    def compute_reward(self, achieved_goal, goal, info=None):
        #print("achieved: {}, \n goal:{}".format(achieved_goal, goal))
        position_tol = 1. # 1 block tolerance to reach a cart or agent goal position
        #print(achieved_goal.shape)
        if achieved_goal.ndim>1:
            #print('array mode')
            bread_goal_achieved = (achieved_goal[:,4:] == goal[:,4:]).all(axis=1)
            #print('bread_goals: {}'.format(bread_goal_achieved))
            agent_cart_positions_achieved = ((np.abs(achieved_goal[:,:4] - goal[:,:4])) < 0.5).all(axis=1)
            #print('agent_goals: {}'.format(agent_cart_positions_achieved))
            rewards = []
            for b_g, a_c_g in zip(bread_goal_achieved, agent_cart_positions_achieved):
                if b_g and a_c_g:
                    rewards.append(1.)
                else:
                    rewards.append(0.)
            #print(rewards)
            return np.array(rewards)
        else:
            bread_goal_achieved = (achieved_goal[4:] == goal[4:]).all()
            agent_cart_positions_achieved = ((np.abs(achieved_goal[:4] - goal[:4])) < 0.5).all()
            if bread_goal_achieved and agent_cart_positions_achieved:
                return np.array([1.])
            else:
                return np.array([0])
        #print("bread: {}".format(bread_goal_achieved))


       
        #print((np.abs(achieved_goal[:3] - goal[:3])))
        #print("agent pos goal: {}".format(((np.abs(achieved_goal[:3] - goal[:3])) < 0.5).all()))


        #d = np.linalg.norm(achieved_goal - goal, ord=2)
        #return -d

    def compute_reward_ddpg(self, state):
        # check whether cart is up
        if state[3] < 286.8 and state[3] > 296.2:
            cart_up = 1
        else:
            cart_up = 0
        reward = state[4:].sum() + self._reward_mixing * cart_up
        return reward



    def render():
     pass

    def close(self):
        pass
