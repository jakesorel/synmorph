#!/bin/bash

((Nfull = "$1"*"$1"*"$1"*"$2"))
((Nfullm1 = "$Nfull" - 1))
((Nper = "$1"*"$2"))
((Nperm1 = "$1"*"$2"-1))
((Njob = "$1"*"$1"))
((Njobm1 = "$1"*"$1"-1))

#rm ../scan_summary/23032022_W01_AVEp0_VEp0_result_log.txt

for i in $(seq 0 "$Njobm1")
do
#    ((start = "$i"*"$Nper"))
#    ((end = "$i"*"$Nper" + "$Nper"))

  sbatch --array [0-"$Nperm1"] run_analysis.sh "$i" "$Nper"
done

#$ sbatch --array [1-Nfullm1] testarray.sh


#for i in $(seq 0 $(($Njob-1)))
#do
##    ./run_simulations.sh "$i" "$Nper"
#    sbatch run_analysis.sh "$i" "$Nper"
#done