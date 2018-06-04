from __future__ import division
import MalmoPython
import json
import time
import sys

class MalmoController(object):

    def __init__(self, mission_xml, port=10000, tick_lengths=50, skip_step=1, desired_mission_time=7):
        self.skip_step = skip_step
        self.tick_lengths = tick_lengths
        self.total_allowed_actions = 10 * desired_mission_time #dependent of skip_step, works if =1

        # Create default Malmo objects:
        self.agent_host = MalmoPython.AgentHost()
        self.my_mission = MalmoPython.MissionSpec(mission_xml, True)
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
        #print("Waiting for the mission to start ", end=' ')
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.001)
            world_state = self.agent_host.getWorldState()
        # dirty solution to wait for client to be stable
        time.sleep(0.5)
        world_state = self.get_world_state(first_state=True)
        obvsText = world_state.observations[-1].text
        observation = json.loads(obvsText) # observation comes in as a JSON string...
        return observation

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

    def step(self, actions):
        # take the action only if mission is still running
        world_state = self.agent_host.peekWorldState()
        if world_state.is_mission_running:
            # take action
            for a in actions:
                self.agent_host.sendCommand(a)

        # wait for the new state
        world_state = self.get_world_state()

        # log errors and control messages
        for error in world_state.errors:
            print error.text

        # detect terminal state
        done = not world_state.is_mission_running
        if not done:
            obvsText = world_state.observations[-1].text
            observation = json.loads(obvsText) # observation comes in as a JSON string...
        else:
            observation = None



        return observation, done