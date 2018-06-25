from __future__ import division
import MalmoPython
import json
import time
import sys
import numpy as np

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

# returns the xml file defining the environment for Malmo
def get_MMC_environment(bread_positions, tick_lengths, skip_step, desired_mission_time):
    total_allowed_actions = 10 * desired_mission_time #dependent of skip_step, works if =1
    # if big overclocking, set display refresh rate to 1
    mod_setting = '' if tick_lengths >= 25 else "<PrioritiseOffscreenRendering>true</PrioritiseOffscreenRendering>"
    print total_allowed_actions
    missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Goal Exploration Process, in Malmo !</Summary>
              </About>

              <ModSettings>
              <MsPerTick>''' + str(tick_lengths) + '''</MsPerTick>
              ''' + mod_setting +'''
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
                  <FileWorldGenerator src="/home/remy/Malmo-0.34.0-Linux-Ubuntu-16.04-64bit_withBoost_Python2.7/Minecraft/run/saves/flowers_v4"/>
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
                  <AgentQuitFromReachingCommandQuota total="'''+ str((2*total_allowed_actions)+1) +'''"/>
                    <VideoProducer>
                      <Width>40</Width>
                      <Height>30</Height>
                    </VideoProducer>
                </AgentHandlers>

              </AgentSection>
            </Mission>'''
    return missionXML


class MalmoController(object):

    def __init__(self, port=10000, tick_lengths=50, skip_step=1, desired_mission_time=7):
        self.skip_step = skip_step
        self.tick_lengths = tick_lengths
        self.total_allowed_actions = 10 * desired_mission_time #dependent of skip_step, works if =1

        # define bread positions in MMC arena
        self.bread_positions = [[293.5,4,436.5],[289.5,4,437.5],[289.5,4,440.5],[291.5,6,442.5],[294.5,6,443.5]]
        self.mission_xml = get_MMC_environment(self.bread_positions, tick_lengths, skip_step, desired_mission_time)
        # Create default Malmo objects:
        self.agent_host = MalmoPython.AgentHost()
        self.my_mission = MalmoPython.MissionSpec(self.mission_xml, True)
        self.my_mission_record = MalmoPython.MissionRecordSpec()

        self.client_pool = MalmoPython.ClientPool()

        print("Attempt to communicate with Minecraft through port %s" % port)
        self.client_pool.add(MalmoPython.ClientInfo( "127.0.0.1", port))


    def start_mission(self):
        # Attempt to start a mission:
        max_retries = 5
        sleep_time = [0.01,0.1,2.,2.,5.]
        for retry in range(max_retries):
            try:
                self.agent_host.startMission(self.my_mission,
                                              self.client_pool, 
                                             self.my_mission_record, 
                                             0,"answer is 42")
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:",e)
                    exit(1)
                else:
                    time.sleep(sleep_time[retry])

        # Loop until mission starts:
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.001)
            world_state = self.agent_host.getWorldState()
        # dirty solution to wait for client to be stable
        time.sleep(0.5)
        world_state = self.get_world_state(first_state=True)
        obvsText = world_state.observations[-1].text
        observation = json.loads(obvsText) # observation comes in as a JSON string...
        self.current_step = 0
        return self.get_state(observation)

    def get_world_state(self, first_state=False):
        # wait till we have got at least one observation or mission has ended
        while True:
            time.sleep(0.001)  # wait for 1ms to not consume entire CPU
            world_state = self.agent_host.peekWorldState()
            #print(world_state.number_of_observations_since_last_state)
            if world_state.number_of_observations_since_last_state > (self.skip_step+1):
                if not first_state:
                    print"DAMMIT, WE LOST %s OBSERVATION" % (world_state.number_of_observations_since_last_state - self.skip_step)
            if world_state.number_of_observations_since_last_state > self.skip_step or not world_state.is_mission_running:
                break
        return self.agent_host.getWorldState()

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
        actions = ["move "+str(actions[0]), "strafe "+str(actions[1])]
        
        self.current_step += 1
        done = False
        #print self.current_step
        # take the action only if mission is still running
        world_state = self.agent_host.peekWorldState()
        if world_state.is_mission_running:
            # take action
            if self.current_step == self.total_allowed_actions + 1: # end of episode
                #last cmd, must teleport to avoid weird glitch with minecart environment
                self.agent_host.sendCommand("tp 293 7 433.5")
                done = True
            else: 
                for a in actions:
                    self.agent_host.sendCommand(a)

                # wait for the new state
                world_state = self.get_world_state()

                # log errors and control messages
                for error in world_state.errors:
                    print error.text
        if not done:
            obvsText = world_state.observations[-1].text
            observation = json.loads(obvsText) # observation comes in as a JSON string...
            state = self.get_state(observation)
            self.previous_state = state
        else:
            state = self.previous_state #last state is state n-1
            # detect terminal state
            #print not world_state.is_mission_running
        
        return state, done