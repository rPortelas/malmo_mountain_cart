#!/bin/bash
malmo_folder="Malmo-0.34.0-Linux-Ubuntu-16.04-64bit_withBoost_Python2.7/"
echo "Bash version ${BASH_VERSION}..."
for ((i = 1; i <= $1; i++)); do
    echo $i
    #xterm -title "app $i" -e "./launchclient & ; sleep 15"
    cd ~/$malmo_folder/Minecraft
    ./launchClient.sh  > malmo_out$i.txt 2>&1 &
    sleep 30
done