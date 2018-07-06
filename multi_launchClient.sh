#!/bin/bash
malmo_folder="malmo0.34_python3.5/"
echo "Bash version ${BASH_VERSION}..."
for ((i = 1; i <= $1; i++)); do
    echo $i
    #xterm -title "app $i" -e "./launchclient & ; sleep 15"
    cd ~/$malmo_folder/Minecraft
    ./launchClient.sh  > malmo_out$i.txt 2>&1 &
    sleep 30
done