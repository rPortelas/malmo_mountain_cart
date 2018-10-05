#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for ((i = 1; i <= $1; i++)); do
    echo $i
    #xterm -title "app $i" -e "./launchclient & ; sleep 15"
    cd ~/$MALMO_DIR/Minecraft
    ./launchClient.sh  > ~/malmo_mountain_cart/malmo_out$i.txt 2>&1 &
    sleep 30
done
