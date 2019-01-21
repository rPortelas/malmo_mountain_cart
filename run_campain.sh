#!/bin/bash

for ((i = 0; i < $1; i++)); do
    #python3 -u imgep_in_emmc_controller.py emmc05_rnd_$i random_flat 5000 5000 0.1 $((10000+($i))) &> rnd_out_$i.txt &
    
    python3 -u imgep_in_emmc_controller.py emmcoldnewint_rmb_$i random_modular 30000 5 0.1 $((10000+(($i)*3))) &> rmb_out_$i.txt &
    python3 -u imgep_in_emmc_controller.py emmcoldnewint_amb_$i active_modular 30000 5 0.1 $((10000+(($i)*3)+1)) &> amb_out_$i.txt &
    python3 -u imgep_in_emmc_controller.py emmcoldnewint_f_rgb_$i random_flat 30000 5 0.1 $((10000+(($i)*3)+2)) &> fmb_out_$i.txt &
    #python3 -u imgep_in_emmc_controller.py emmcbt_rnd_$i random_flat 25000 25000 0.1 $((10000+(($i-3)*3)+3)) &> rnd_out_$i.txt &
    #> run$i_output.txt 2>&1 &
done
