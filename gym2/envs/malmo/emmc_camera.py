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

PICKAXE_POS = [292, 436]
D_TOOL_POS = [290, 436]
LOG = False

def safeStartMission(agent_host, my_mission, my_client_pool, my_mission_record, role, expId):
    used_attempts = 0
    max_attempts = 5
    print("Calling startMission for role", role)
    while True:
        try:
            # Attempt start:
            agent_host.startMission(my_mission, my_client_pool, my_mission_record, role, expId)
            break
        except MalmoPython.MissionException as e:
            errorCode = e.details.errorCode
            if errorCode == MalmoPython.MissionErrorCode.MISSION_SERVER_WARMING_UP:
                print("Server not quite ready yet - waiting...")
                time.sleep(2)
            elif errorCode == MalmoPython.MissionErrorCode.MISSION_INSUFFICIENT_CLIENTS_AVAILABLE:
                print("Not enough available Minecraft instances running.")
                used_attempts += 1
                if used_attempts < max_attempts:
                    print("Will wait in case they are starting up.", max_attempts - used_attempts, "attempts left.")
                    time.sleep(2)
            elif errorCode == MalmoPython.MissionErrorCode.MISSION_SERVER_NOT_FOUND:
                print("Server not found - has the mission with role 0 been started yet?")
                used_attempts += 1
                if used_attempts < max_attempts:
                    print("Will wait and retry.", max_attempts - used_attempts, "attempts left.")
                    time.sleep(2)
            else:
                print("Other error:", e.message)
                print("Waiting will not help here - bailing immediately.")
                exit(1)
        if used_attempts == max_attempts:
            print("All chances used up - bailing now.")
            exit(1)
    print("startMission called okay.")

def safeWaitForStart(agent_hosts):
    print("Waiting for the mission to start", end=' ')
    start_flags = [False for a in agent_hosts]
    start_time = time.time()
    time_out = 120  # Allow a two minute timeout.
    while not all(start_flags) and time.time() - start_time < time_out:
        states = [a.peekWorldState() for a in agent_hosts]
        start_flags = [w.has_mission_begun for w in states]
        errors = [e for w in states for e in w.errors]
        if len(errors) > 0:
            print("Errors waiting for mission start:")
            for e in errors:
                print(e.text)
            print("Bailing now.")
            exit(1)
        time.sleep(0.1)
        print(".", end=' ')
    if time.time() - start_time >= time_out:
        print("Timed out while waiting for mission to start - bailing.")
        exit(1)
    print()
    print("Mission has started.")


def get_MMC_environment(tick_lengths, total_allowed_actions):
    # if big overclocking, set display refresh rate to 1
    mod_setting = '' if tick_lengths >= 25 else "<PrioritiseOffscreenRendering>true</PrioritiseOffscreenRendering>"
    max_time = str((50 / tick_lengths) * 25000)
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
                  <FlatWorldGenerator forceReset="true" />
                  <DrawingDecorator>

                    <!-- Draw floor -->
                    <DrawCuboid x1="286" y1="3" z1="429" x2="296" y2="3" z2="445" type="air"/>
                    <DrawCuboid x1="286" y1="3" z1="429" x2="296" y2="2" z2="445" type="log" />

                    <!-- Draw arena long side -->
                    <DrawCuboid x1="287" y1="4" z1="431" x2="295" y2="6" z2="445" type="air"/>
                    <DrawCuboid x1="287" y1="4" z1="430" x2="295" y2="6" z2="444" type="planks"/>
                    <DrawCuboid x1="289" y1="4" z1="431" x2="293" y2="6" z2="443" type="air"/>
                    <DrawCuboid x1="288" y1="4" z1="431" x2="294" y2="6" z2="439" type="air"/>


                    <!-- Draw diamond blocks -->
                    <DrawCuboid x1="290" y1="5" z1="441" x2="292" y2="5" z2="441" type="diamond_ore"/>
                    <!-- fill row -->
                     <DrawBlock x="293" y="5" z="441" type="diamond_ore" />
                      <DrawBlock x="289" y="5" z="441" type="diamond_ore" />
                     <!-- obsidian used as markers to detect diamond line -->
                    <DrawBlock x="294" y="5" z="441" type="obsidian" />

                    <!-- Draw rail tracks -->
                    <DrawLine x1="288" y1="4" z1="443" x2="294" y2="4" z2="443" type="air"/>

                    <DrawLine x1="286" y1="5" z1="443" x2="296" y2="5" z2="443" type="planks"/>
                    <DrawLine x1="287" y1="5" z1="443" x2="295" y2="5" z2="443" type="air"/>

                    <DrawLine x1="285" y1="6" z1="443" x2="297" y2="6" z2="443" type="gold_block"/>
                    <DrawLine x1="286" y1="6" z1="443" x2="296" y2="6" z2="443" type="air"/>


                    <DrawBlock x="287" y="5" z="443" type="rail"/>
                    <DrawBlock x="286" y="6" z="443" type="rail"/>
                    <DrawBlock x="295" y="5" z="443" type="rail"/>
                    <DrawBlock x="296" y="6" z="443" type="rail"/>
                    <DrawLine x1="288" y1="4" z1="443" x2="294" y2="4" z2="443" type="rail"/>


                    <DrawEntity x="291.5" y="7" z="443" type="MinecartRideable"/>

                    <!-- Draw tools cage -->
                    <DrawCuboid x1="289" y1="5" z1="437" x2="293" y2="5" z2="437" type="bedrock"/>
                    <DrawCuboid x1="289" y1="4" z1="437" x2="293" y2="4" z2="437" type="planks"/>
                    <DrawCuboid x1="289" y1="4" z1="435" x2="293" y2="4" z2="435" type="planks"/>
                    <DrawBlock x="291" y="4" z="436" type="planks" />

                     <!-- Draw starting cage -->
                    <!--<DrawCuboid x1="289" y1="4" z1="435" x2="289" y2="4" z2="433" type="bedrock"/>-->
                    <!--<DrawCuboid x1="293" y1="4" z1="435" x2="293" y2="4" z2="433" type="bedrock"/>-->
                    <!--<DrawBlock x="291" y="4" z="436" type="bedrock" />-->

                    <DrawBlock x="292" y="3" z="434" type="water" />
                    <DrawBlock x="293" y="3" z="434" type="water" />
                    <DrawBlock x="292" y="3" z="433" type="water" />
                    <DrawBlock x="292" y="3" z="432" type="water" />
                    <DrawBlock x="290" y="3" z="434" type="water" />
                    <DrawBlock x="289" y="3" z="434" type="water" />
                    <DrawBlock x="290" y="3" z="433" type="water" />
                    <DrawBlock x="290" y="3" z="432" type="water" />
                    <DrawBlock x="288" y="3" z="432" type="water" />
                    <DrawBlock x="288" y="3" z="431" type="water" />
                    <DrawBlock x="294" y="3" z="431" type="water" />
                    <DrawBlock x="294" y="3" z="432" type="water" />

                    <!-- Draw holes -->
                    <DrawBlock x="294" y="3" z="439" type="water" />
                    <DrawBlock x="288" y="3" z="439" type="water" />




                    <!-- Draw tools -->
                    <DrawItem x="''' + str(PICKAXE_POS[0]) + '''" y="4" z="''' + str(PICKAXE_POS[1]) + '''" type="diamond_pickaxe"/>
                    <DrawItem x="''' + str(D_TOOL_POS[0]) + '''" y="4" z="''' + str(D_TOOL_POS[1]) + '''" type="diamond_shovel"/>

                    <!-- Draw distractors -->
                    <DrawBlock x="291" y="6" z="444" type="grass" />
                    <DrawBlock x="294" y="6" z="440" type="grass" />
                    <DrawBlock x="288" y="6" z="440" type="grass" />
                    <DrawBlock x="291" y="7" z="444" type="red_flower" />
                    <DrawBlock x="294" y="7" z="440" type="red_flower" />
                    <DrawBlock x="288" y="7" z="440" type="red_flower" />

                  </DrawingDecorator>
                  <ServerQuitWhenAnyAgentFinishes/>
                  <ServerQuitFromTimeUp description="" timeLimitMs="''' + max_time + '''"/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>FlowersBot</Name>
                <AgentStart>
                  <Placement x="291.5" y="4.2" z="432.5" yaw="0"/>
                  <Inventory></Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromGrid>
                      <Grid name="grid">
                        <min x="-4" y="1" z="1"/>
                        <max x="5" y="1" z="3"/>
                      </Grid>
                  </ObservationFromGrid>
                  <ObservationFromNearbyEntities>
                  <Range name="entities" xrange="15" yrange="15" zrange="15"/>
                  </ObservationFromNearbyEntities>
                  <AbsoluteMovementCommands/>
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                  <MissionQuitCommands/>
                  <AgentQuitFromReachingCommandQuota total="''' + str((3 * total_allowed_actions) + 1) + '''"/>
                  <VideoProducer>
                    <Width>400</Width>
                    <Height>300</Height>
                  </VideoProducer>
                </AgentHandlers>
                </AgentSection>
                
                <AgentSection mode="Creative">
                <Name>TheWatcher</Name>
                <AgentStart>
                  <Placement x="291.5" y="4.2" z="433.5" pitch="90"/>
                </AgentStart>
                <AgentHandlers>
                  <ContinuousMovementCommands turnSpeedDegs="360"/>
                  <MissionQuitCommands/>
                  <VideoProducer>
                    <Width>640</Width>
                    <Height>640</Height>
                  </VideoProducer>
                </AgentHandlers>
              </AgentSection>

            </Mission>'''
    return missionXML


class EMMCWithCam(gym2.Env):
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
        self.mission_start_sleep = 0.1
        self.mission_xml = get_MMC_environment(tick_lengths, self.total_allowed_actions)
        # Create default Malmo objects:
        self.agent_host = MalmoPython.AgentHost()
        self.agent_host_observer = MalmoPython.AgentHost()
        self.my_mission = MalmoPython.MissionSpec(self.mission_xml, True)

        self.client_pool = MalmoPython.ClientPool()

        # print("Attempt to communicate with Minecraft")
        # enable the use of up to 21 parallel malmo mountain carts
        # for i in range(20):
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
        self.my_mission_record = MalmoPython.MissionRecordSpec()
        self.client_pool = MalmoPython.ClientPool()
        for i in range(20):
            self.client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10001 + i))

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
            self.agent_host_observer.sendCommand("quit")
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
                time.sleep(5)
                self.agent_host_observer.startMission(self.my_mission,
                                             self.client_pool,
                                             self.my_mission_record,
                                             0, "answer is kljmlk42")

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
        world_state_obs = self.agent_host_observer.peekWorldState()
        while (not world_state.has_mission_begun) and (not world_state_obs.has_mission_begun):
            time.sleep(0.001)
            world_state = self.agent_host.peekWorldState()
            world_state_obs = self.agent_host_obs.peekWorldState()
        time.sleep(self.mission_start_sleep)  # dirty to way to wait for server stability

        world_state = self.get_world_state(first_state=True)
        obvsText = world_state.observations[-1].text
        observation = json.loads(obvsText)  # observation comes in as a JSON string...
        self.current_step = 0

        self.last_block_state = [-1., -1., -1., -1., -1.]

        # self.desired_goal = self.sample_goal()
        state = self.get_state(observation)
        # reward = self.compute_reward(state, self.desired_goal)

        obs = dict(observation=state,
                   achieved_goal=state,
                   desired_goal=None)

        return obs

    def extract_block_state(self, obs):
        if self.current_step <= 5:
            # avoid using grid observation in (unstable) mission starting part
            return self.last_block_state
        grid = np.array(obs['grid']).reshape(3, 10)
        marker_pos = np.argwhere(grid == 'obsidian')
        if len(marker_pos) == 0:
            agent = obs['entities'][0]
            assert (agent['name'] == 'FlowersBot')
            agent_pos = [agent['x'], agent['y'], agent['z']]
            # print('WARNING obsidian marker not detected !!! grid: {}, step: {}, agent_pos: {}'.format(grid, self.current_step, agent_pos))
            return self.last_block_state
        start_x, start_y = marker_pos[0][0], marker_pos[0][1]
        diamond_blocks = grid[start_x, start_y - 5:start_y]
        block_state = [-1. if v == 'diamond_ore' else 1. for v in diamond_blocks]
        if not len(block_state) == 5:  # not in front of diamond yet
            return self.last_block_state
        self.last_block_state = block_state
        return self.last_block_state

    def get_state(self, obs):
        # print(obs)
        blocks = self.extract_block_state(obs)
        agent = obs['entities'][0]
        assert (agent['name'] == 'FlowersBot')
        agent_pos = [agent['x'], agent['z']]
        pickaxe_pos = None
        shovel_pos = None
        for e in obs['entities']:
            if e['name'] == 'diamond_pickaxe':
                pickaxe_pos = [e['x'], e['z']]
            if e['name'] == 'diamond_shovel':
                shovel_pos = [e['x'], e['z']]
            if e['name'] == 'MinecartRideable':
                cart_x = [e['x']]
        if pickaxe_pos == None:  # If not in arena then agents has it
            pickaxe_pos = agent_pos
        if shovel_pos == None:
            shovel_pos = agent_pos

        # print(agent_pos + pickaxe_pos + shovel_pos + blocks + cart_x)
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