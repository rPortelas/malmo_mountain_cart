#!/bin/bash
malmo_folder="Malmo-0.34.0-Linux-Ubuntu-16.04-64bit_withBoost_Python2.7/"
echo "Bash version ${BASH_VERSION}..."
for ((i = 0; i < $1; i++)); do
    echo $i
    #xterm -title "app $i" -e "./launchclient & ; sleep 15"
    #./launchClient.sh  > malmo_out$i.txt 2>&1 &
    #port=$((10000 + $i))
    port=10000
    echo hello
    echo $port
    python imgep_in_mmc_controller.py test$i random_flat 10000 1000 0.1 $port &
    #> run$i_output.txt 2>&1 &
done
