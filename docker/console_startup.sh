#!/bin/bash
/dockerstartup/vnc_startup.sh echo "Starting Malmo Minecraft Mod"
cd /home/malmo
# Launch mimecraft (which may take several minutes first time)
#python3 -c "import malmo.minecraftbootstrap;malmo.minecraftbootstrap.launch_minecraft()"
cd MalmoPlatform/Minecraft/
echo "WILL BE USING PORT $1"
xvfb-run -a -e /dev/stdout -s '-screen 0 1400x900x24' ./launchClient.sh -port $1
#echo "Starting jupyter TEST MALMO"
#python3 test_malmo.py
#jupyter notebook --ip 0.0.0.0 --no-browser
