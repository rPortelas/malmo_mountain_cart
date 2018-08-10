#!/bin/bash
for ((i = 0; i < $1; i++)); do
    port=10000
    python imgep_in_mmc_controller.py test$i random_flat 10000 1000 0.1 $port &
    #> run$i_output.txt 2>&1 &
done
