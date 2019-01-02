import gym2
from gym2 import spaces
from gym2.utils import seeding
import numpy as np
import malmo.MalmoPython as MalmoPython
import json
import time
import random
import sys
import getpass
import os
from utils.gep_utils import Bounds, unscale_vector

PICKAXE_POS = [292,436]
D_TOOL_POS = [290,436]
LOG = False
def get_MMC_environment(tick_lengths, total_allowed_actions):
    # if big overclocking, set display refresh rate to 1
    mod_setting = '' if tick_lengths >= 25 else "<PrioritiseOffscreenRendering>true</PrioritiseOffscreenRendering>"
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
                  <FlatWorldGenerator forceReset="false" />
                  <DrawingDecorator>
                  
                    <!-- Draw floor -->
                    <DrawCuboid x1="285" y1="3" z1="431" x2="302" y2="3" z2="446" type="air"/>
                    <DrawCuboid x1="285" y1="3" z1="431" x2="302" y2="3" z2="446" type="bedrock" />
                    
                    <!-- Draw arena long side -->
                    <DrawCuboid x1="287" y1="4" z1="432" x2="295" y2="6" z2="445" type="air"/>
                    <DrawCuboid x1="287" y1="4" z1="432" x2="295" y2="6" z2="445" type="bedrock"/>
                    <DrawCuboid x1="289" y1="4" z1="433" x2="293" y2="6" z2="443" type="air"/>
                    <DrawCuboid x1="288" y1="4" z1="433" x2="294" y2="6" z2="439" type="air"/>
                    
                    <!-- Draw arena width side -->
                    <DrawCuboid x1="289" y1="6" z1="445" x2="293" y2="6" z2="445" type="air"/>
                    <DrawCuboid x1="297" y1="6" z1="445" x2="285" y2="9" z2="445" type="bedrock"/> 
                    
                    <!-- Draw diamond blocks -->
                    <DrawCuboid x1="290" y1="5" z1="441" x2="292" y2="5" z2="441" type="coal_ore"/>
                    <!-- fill row -->
                     <DrawBlock x="293" y="5" z="441" type="coal_ore" />
                      <DrawBlock x="289" y="5" z="441" type="coal_ore" />
                     <!-- obsidian used as markers to detect diamond line -->
                    <DrawBlock x="294" y="5" z="441" type="obsidian" />
                    
                    <!-- Draw rail tracks -->
                    <DrawLine x1="288" y1="4" z1="443" x2="294" y2="4" z2="443" type="air"/>

                    <DrawLine x1="286" y1="5" z1="443" x2="296" y2="5" z2="443" type="bedrock"/>
                    <DrawLine x1="287" y1="5" z1="443" x2="295" y2="5" z2="443" type="air"/>

                    <DrawLine x1="285" y1="6" z1="443" x2="297" y2="6" z2="443" type="gold_block"/>
                    <DrawLine x1="286" y1="6" z1="443" x2="296" y2="6" z2="443" type="air"/>

  
                    <DrawBlock x="287" y="5" z="443" type="rail"/>
                    <DrawBlock x="286" y="6" z="443" type="rail"/>
                    <DrawBlock x="295" y="5" z="443" type="rail"/>
                    <DrawBlock x="296" y="6" z="443" type="rail"/>
                    <DrawLine x1="288" y1="4" z1="443" x2="294" y2="4" z2="443" type="rail"/>
                                      
                    
                    <DrawEntity x="291.5" y="7" z="443" type="MinecartRideable"/>
                    
                    <!-- Draw starting cage -->
                    <DrawCuboid x1="289" y1="4" z1="437" x2="293" y2="5" z2="437" type="bedrock"/>
                    <DrawCuboid x1="289" y1="4" z1="435" x2="293" y2="4" z2="435" type="bedrock"/>
                    <DrawBlock x="291" y="4" z="436" type="bedrock" />

                  
                    <!-- Draw tools -->
                    <DrawItem x="''' + str(PICKAXE_POS[0]) + '''" y="4" z="'''+ str(PICKAXE_POS[1]) + '''" type="golden_pickaxe"/>
                    <DrawItem x="''' + str(D_TOOL_POS[0]) + '''" y="4" z="'''+ str(D_TOOL_POS[1]) + '''" type="diamond_shovel"/>

                  </DrawingDecorator>
                  <ServerQuitWhenAnyAgentFinishes/>
                  <ServerQuitFromTimeUp description="" timeLimitMs="25000"/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>FlowersBot</Name>
                <AgentStart>
                  <Placement x="291.5" y="4.2" z="433.5" yaw="0"/>
                  <Inventory></Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromGrid>
                      <Grid name="grid">
                        <min x="-6" y="1" z="-4"/>
                        <max x="6" y="1" z="8"/>
                      </Grid>
                  </ObservationFromGrid>
                  <ObservationFromNearbyEntities>
                  <Range name="entities" xrange="15" yrange="15" zrange="15"/>
                  </ObservationFromNearbyEntities>
                  <AbsoluteMovementCommands/>
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                  <MissionQuitCommands/>
                  <AgentQuitFromReachingCommandQuota total="''' + str((3 * total_allowed_actions)+1) + '''"/>
                  <VideoProducer>
                    <Width>40</Width>
                    <Height>30</Height>
                  </VideoProducer>
                </AgentHandlers>

              </AgentSection>
            </Mission>'''
    return missionXML

class ExtendedMalmoMountainCart(gym2.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30,
    }

    def __init__(self, port=10000, tick_lengths=15, skip_step=4, desired_mission_time=8, sparse=False,
                 reward_mixing=20):
        self.skip_step = skip_step
        self.tick_lengths = tick_lengths
        self.total_allowed_actions = int((20 / (skip_step + 1)) * desired_mission_time)
        self._sparse = sparse
        self._reward_mixing = reward_mixing
        self.mission_start_sleep = 0.2
        self.mission_xml = get_MMC_environment(tick_lengths, self.total_allowed_actions)
        # Create default Malmo objects:
        self.agent_host = MalmoPython.AgentHost()
        self.my_mission = MalmoPython.MissionSpec(self.mission_xml, True)

        self.client_pool = MalmoPython.ClientPool()

        #print("Attempt to communicate with Minecraft")
        # enable the use of up to 21 parallel malmo mountain carts
        #for i in range(20):
        #    self.client_pool.add(MalmoPython.ClientInfo("127.0.0.1", port + i))

        n_act = 3
        n_obs = 12

        self.action_space = spaces.Box(low=-np.ones(n_act), high=np.ones(n_act))
        self.observation_space = spaces.Box(low=-np.repeat([np.array([np.inf])], n_obs),
                                            high=np.repeat([np.array([np.inf])], n_obs))

        self.current_step = 0
        self.reset_nb = 0
        self.seed()

    # call this method to change default parameters
    def my_init(self, port=None, tick_lengths=15, skip_step=4, desired_mission_time=8):
        self.skip_step = skip_step
        self.tick_lengths = tick_lengths
        self.total_allowed_actions = int((20 / (skip_step + 1)) * desired_mission_time)
        self.mission_xml = get_MMC_environment(tick_lengths, self.total_allowed_actions)
        # Create default Malmo objects:
        self.agent_host = MalmoPython.AgentHost()
        self.my_mission = MalmoPython.MissionSpec(self.mission_xml, True)
        self.my_mission_record = MalmoPython.MissionRecordSpec()

        if port is not None:
            self.client_pool = MalmoPython.ClientPool()
            self.client_pool.add(MalmoPython.ClientInfo("127.0.0.1", port))

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return seed

    def sample_goal(self):
        goal = np.random.random(9) * 2 - 1
        env_scaled_goal = unscale_vector(goal, np.array(self.b.get_bounds(self.state_names)))
        env_scaled_goal[4:] = np.array([0 if b < 0.5 else 1 for b in env_scaled_goal[4:]])
        # print("goal sampled: {}".format(env_scaled_goal))
        return env_scaled_goal

    def get_world_state(self, first_state=False):
        # wait till we have got at least one observation or mission has ended
        while True:
            time.sleep(0.001)  # wait for 1ms to not consume entire CPU
            world_state = self.agent_host.peekWorldState()
            # print(world_state.number_of_observations_since_last_state)
            if world_state.number_of_observations_since_last_state > (self.skip_step + 1):
                if not first_state:
                    print("DAMMIT, WE LOST %s OBSERVATION, step %s" % (
                                world_state.number_of_observations_since_last_state - self.skip_step, self.current_step))
            if world_state.number_of_observations_since_last_state > self.skip_step or not world_state.is_mission_running:
                break
        return self.agent_host.getWorldState()

    def reset(self, goal=None):
        world_state = self.agent_host.peekWorldState()
        if LOG: print("resetting, previous mission running ?: {}".format(world_state.is_mission_running))
        if world_state.is_mission_running:
            self.agent_host.sendCommand("quit")
            while world_state.is_mission_running:
                print('waiting mission abortion...')
                time.sleep(0.01)
                world_state = self.agent_host.peekWorldState()
            print('aborted')

        # dirty solution to wait for client to be stable
        time.sleep(0.08)

        # Attempt to start a mission:
        max_retries = 7
        sleep_time = [0.01, 0.1, 0.2, 0.4, 2., 5., 5.]
        for retry in range(max_retries):
            try:
                if LOG: print('trying to start mission')
                # world_state = self.agent_host.peekWorldState()
                if LOG: print("SHOULD BE FALSE: {}".format(world_state.is_mission_running))
                self.agent_host.startMission(self.my_mission,
                                             self.client_pool,
                                             self.my_mission_record,
                                             0, "answer is 42")

                break
            except RuntimeError as e:
                if LOG: print('failed')
                if retry == max_retries - 1:
                    print("!!!!!!!!!!!!!!!!!!!!!!Error starting mission: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!", e)
                    exit(1)
                else:
                    time.sleep(sleep_time[retry])

        # Loop until mission starts:
        if LOG: print('Loop until mission starts')
        world_state = self.agent_host.peekWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.001)
            world_state = self.agent_host.peekWorldState()

        time.sleep(self.mission_start_sleep) #dirty to way to wait for server stability

        world_state = self.get_world_state(first_state=True)
        obvsText = world_state.observations[-1].text
        observation = json.loads(obvsText)  # observation comes in as a JSON string...
        self.current_step = 0

        #self.desired_goal = self.sample_goal()
        state = self.get_state(observation)
        #reward = self.compute_reward(state, self.desired_goal)

        obs = dict(observation=state,
                   achieved_goal=state,
                   desired_goal=None)
        return obs

    def extract_block_state(self, obs):
        if self.current_step <= 2:
            # avoid using grid observation in (unstable) mission starting part
            return [-1., -1., -1., -1., -1.]
        grid = np.array(obs['grid']).reshape(13, 13)
        marker_pos = np.argwhere(grid == 'obsidian')
        if len(marker_pos) == 0:
            agent = obs['entities'][0]
            assert (agent['name'] == 'FlowersBot')
            agent_pos = [agent['x'], agent['y'], agent['z']]
            print('WARNING obsidian marker not detected !!! grid: {}, step: {}, agent_pos: {}'.format(grid, self.current_step, agent_pos))
            return [-1., -1., -1., -1., -1.]
        start_x, start_y = marker_pos[0][0], marker_pos[0][1]
        diamond_blocks = grid[start_x,start_y-5:start_y]
        return [-1. if v=='coal_ore' else 1. for v in diamond_blocks]

    def get_state(self, obs):
        #print(obs)
        blocks = self.extract_block_state(obs)
        agent = obs['entities'][0]
        assert(agent['name'] == 'FlowersBot')
        agent_pos = [agent['x'], agent['z']]
        pickaxe_pos = None
        shovel_pos = None
        for e in obs['entities']:
            if e['name'] == 'golden_pickaxe':
                pickaxe_pos = [e['x'], e['z']]
            if e['name'] == 'diamond_shovel':
                shovel_pos = [e['x'], e['z']]
            if e['name'] == 'MinecartRideable':
                cart_x = [e['x']]
        if pickaxe_pos == None: #If not in arena then agents has it
            pickaxe_pos = agent_pos
        if shovel_pos == None:
            shovel_pos = agent_pos

        #print(agent_pos + pickaxe_pos + shovel_pos + blocks + cart_x)
        return np.array(agent_pos + pickaxe_pos + shovel_pos + blocks + cart_x)

    def step(self, actions):
        if LOG: print('time:{}, starting step number {}, acts: {}'.format(time.time(), self.current_step, actions))
        if self.current_step == self.total_allowed_actions:
            print('Trying to take action in finished episode')
            return 0
        # format actions for environment
        actions = ["move " + str(actions[0]), "strafe " + str(actions[1]), "attack " + str(1 if actions[2] >= 0 else 0)]
        self.current_step += 1
        done = False
        # print self.current_step
        # take the action only if mission is still running
        world_state = self.agent_host.peekWorldState()
        # print(self.current_step)
        if world_state.is_mission_running:
            # take action
            for a in actions:
                self.agent_host.sendCommand(a)

            # wait for the new state
            if self.current_step == 1:  # first time we acted, discard missed observation warning
                world_state = self.get_world_state(first_state=True)
            else:
                world_state = self.get_world_state()

            # log errors and control messages
            for error in world_state.errors:
                print(error.text)

            if world_state.is_mission_running:
                obvsText = world_state.observations[-1].text
                observation = json.loads(obvsText)  # observation comes in as a JSON string...
                state = self.get_state(observation)
                self.previous_state = state
            else:
                print('SHOULD NOT HAPPEN, MISSION ENDED AT STEP {} BEFORE THEORETICAL END'.format(self.current_step))
                state = self.previous_state

            if self.current_step == self.total_allowed_actions:  # end of episode
                # send quit action
                self.agent_host.sendCommand("quit")
                world_state = self.agent_host.peekWorldState()
                while world_state.is_mission_running:
                    if LOG: print('waiting for end of mission')
                    time.sleep(0.01)
                    world_state = self.agent_host.peekWorldState()
                done = True
        else:
            print('MISSION ABORTED BEFORE END OF EP, step:{}'.format(self.current_step))
            done = True

        obs = dict(observation=state,
                   achieved_goal=state,
                   desired_goal=None)
        return obs, 0, done, {}

    def compute_reward(self, achieved_goal, goal, info=None):
        # print("achieved: {}, \n goal:{}".format(achieved_goal, goal))
        position_tol = 1.  # 1 block tolerance to reach a cart or agent goal position
        # print(achieved_goal.shape)
        if achieved_goal.ndim > 1:
            # print('array mode')
            bread_goal_achieved = (achieved_goal[:, 4:] == goal[:, 4:]).all(axis=1)
            # print('bread_goals: {}'.format(bread_goal_achieved))
            agent_cart_positions_achieved = ((np.abs(achieved_goal[:, :4] - goal[:, :4])) < 1.5).all(axis=1)
            # print('agent_goals: {}'.format(agent_cart_positions_achieved))
            rewards = []
            for b_g, a_c_g in zip(bread_goal_achieved, agent_cart_positions_achieved):
                if b_g and a_c_g:
                    rewards.append(1.)
                else:
                    rewards.append(0.)
            # print(rewards)
            return np.array(rewards)
        else:
            bread_goal_achieved = (achieved_goal[4:] == goal[4:]).all()
            agent_cart_positions_achieved = ((np.abs(achieved_goal[:4] - goal[:4])) < 1.5).all()
            if bread_goal_achieved and agent_cart_positions_achieved:
                return np.array([1.])
            else:
                return np.array([0])

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
        print('closing instance')
        world_state = self.agent_host.peekWorldState()
        print("IS RUNNING ?: {}".format(world_state.is_mission_running))
        if world_state.is_mission_running:
            self.agent_host.sendCommand("quit")
            while world_state.is_mission_running:
                print('waiting mission abortion...')
                time.sleep(0.01)
                world_state = self.agent_host.peekWorldState()
            print('aborted')