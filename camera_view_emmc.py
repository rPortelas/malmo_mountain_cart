from __future__ import print_function
from __future__ import division
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Test of multi-agent missions - runs a number of agents in a shared environment.

from builtins import range
from past.utils import old_div
import malmo.MalmoPython as MalmoPython
import json
import logging
import math
import os
import random
import sys
import time
import re
import uuid
from collections import namedtuple
from operator import add

EntityInfo = namedtuple('EntityInfo', 'x, y, z, name')

# Create one agent host for parsing:
agent_hosts = [MalmoPython.AgentHost()]

# Parse the command-line options:
agent_hosts[0].addOptionalFlag( "debug,d", "Display debug information.")
agent_hosts[0].addOptionalIntArgument("agents,n", "Number of agents to use, including observer.", 4)

PICKAXE_POS = [292, 436]
D_TOOL_POS = [290, 436]
try:
    agent_hosts[0].parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_hosts[0].getUsage())
    exit(1)
if agent_hosts[0].receivedArgument("help"):
    print(agent_hosts[0].getUsage())
    exit(0)

DEBUG = agent_hosts[0].receivedArgument("debug")
INTEGRATION_TEST_MODE = agent_hosts[0].receivedArgument("test")
agents_requested = agent_hosts[0].getIntArgument("agents")
NUM_AGENTS = 1 # Will be NUM_AGENTS robots running around, plus one static observer.
NUM_MOBS = NUM_AGENTS * 2
NUM_ITEMS = NUM_AGENTS * 2

# Create the rest of the agent hosts - one for each robot, plus one to give a bird's-eye view:
agent_hosts += [MalmoPython.AgentHost() for x in range(1, NUM_AGENTS + 1) ]

# Set up debug output:
for ah in agent_hosts:
    ah.setDebugOutput(DEBUG)    # Turn client-pool connection messages on/off.

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

def agentName(i):
    return "Robot#" + str(i + 1)

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

def drawMobs():
    xml = ""
    for i in range(NUM_MOBS):
        x = str(random.randint(-17,17))
        z = str(random.randint(-17,17))
        xml += '<DrawEntity x="' + x + '" y="214" z="' + z + '" type="Zombie"/>'
    return xml

def drawItems():
    xml = ""
    for i in range(NUM_ITEMS):
        x = str(random.randint(-17,17))
        z = str(random.randint(-17,17))
        xml += '<DrawItem x="' + x + '" y="224" z="' + z + '" type="apple"/>'
    return xml

def getXML(reset):
    tick_lengths = 50
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
                        
                        <DrawLine x1="287" y1="6" z1="442" x2="295" y2="6" z2="442" type="air"/>




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
                      <ServerQuitFromTimeUp description="" timeLimitMs="''' + str(99999) + '''"/>
                    </ServerHandlers>
                  </ServerSection>

                  <AgentSection mode="Survival">
                    <Name>Learner</Name>
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
                      <VideoProducer>
                        <Width>400</Width>
                        <Height>300</Height>
                      </VideoProducer>
                    </AgentHandlers>
                    </AgentSection>

                    <AgentSection mode="Creative">
                    <Name>TheWatcher</Name>
                    <AgentStart>
                      <Placement x="291.5" y="4.2" z="453.5" pitch="0"/>
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

# Set up a client pool.
# IMPORTANT: If ANY of the clients will be on a different machine, then you MUST
# make sure that any client which can be the server has an IP address that is
# reachable from other machines - ie DO NOT SIMPLY USE 127.0.0.1!!!!
# The IP address used in the client pool will be broadcast to other agents who
# are attempting to find the server - so this will fail for any agents on a
# different machine.
client_pool = MalmoPython.ClientPool()
for x in range(10000, 10000 + NUM_AGENTS + 1):
    client_pool.add( MalmoPython.ClientInfo('127.0.0.1', x) )

num_missions = 5 if INTEGRATION_TEST_MODE else 30000
for mission_no in range(1, num_missions+1):
    print("Running mission #" + str(mission_no))
    # Create mission xml - use forcereset if this is the first mission.
    my_mission = MalmoPython.MissionSpec(getXML("true" if mission_no == 1 else "false"), True)

    # Generate an experiment ID for this mission.
    # This is used to make sure the right clients join the right servers -
    # if the experiment IDs don't match, the startMission request will be rejected.
    # In practice, if the client pool is only being used by one researcher, there
    # should be little danger of clients joining the wrong experiments, so a static
    # ID would probably suffice, though changing the ID on each mission also catches
    # potential problems with clients and servers getting out of step.

    # Note that, in this sample, the same process is responsible for all calls to startMission,
    # so passing the experiment ID like this is a simple matter. If the agentHosts are distributed
    # across different threads, processes, or machines, a different approach will be required.
    # (Eg generate the IDs procedurally, in a way that is guaranteed to produce the same results
    # for each agentHost independently.)
    experimentID = str(uuid.uuid4())

    for i in range(len(agent_hosts)):
        safeStartMission(agent_hosts[i], my_mission, client_pool, MalmoPython.MissionRecordSpec(), i, experimentID)

    safeWaitForStart(agent_hosts)

    time.sleep(500)