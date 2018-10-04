#!/bin/bash

for ((i = 0; i < $1; i++)); do
    python3 imgep_in_mmc_controller.py test_rmb_$i random_modular &
    python3 imgep_in_mmc_controller.py test_amb_$i active_modular &
    python3 imgep_in_mmc_controller.py test_f_rgb_$i random_flat &
    python3 imgep_in_mmc_controller.py test_random_$i random_flat 10000 10000 &
    #> run$i_output.txt 2>&1 &
done

# for ((i = 0; i < $1; i++)); do
#     python3 imgep_in_mmc_controller.py test_amb_$i active_modular &
#     #> run$i_output.txt 2>&1 &
# done
