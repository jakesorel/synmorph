#!/bin/bash

((Nfull = "$1"*"$1"*"$1"*"$2"))
((Nper = "$1"*"$2"))
((Njob = "$1"*"$1"))

#rm ../scan_summary/23032022_W01_AVEp0_VEp0_result_log.txt

for i in $(seq 0 $(($Njob-1)))
do
#    ./run_simulations.sh "$i" "$Nper"
    sbatch run_analysis.sh "$i" "$Nper"
done