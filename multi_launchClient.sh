#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for ((i = 1; i <= $1; i++)); do
    echo $i
    #xterm -title "app $i" -e "./launchclient & ; sleep 15"
    ./launchClient.sh  > malmo_out$i.txt 2>&1 &
    sleep 15
done