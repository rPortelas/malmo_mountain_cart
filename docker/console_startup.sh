#!/bin/bash
cd /home/malmo
cd MalmoPlatform/Minecraft/
echo "WILL BE USING PORT $1"
xpra start :100
export DISPLAY=:100
./launchClient.sh -port $1