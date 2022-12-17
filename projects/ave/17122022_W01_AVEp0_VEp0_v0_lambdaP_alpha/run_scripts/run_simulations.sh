#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=00:01:00   # walltime
#SBATCH -J "AVE_simulation_loader"   # job name
#SBATCH --output=../bash_out/output_loader.out
#SBATCH --error=../bash_out/error_loader.out
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --mem=5M

((j = "$2"*"$1"))
((terminate = "$2"*"$1" + "$2"))

sbatch run_simulation.sh "$j" "$terminate"
#for i in $(seq 0 "$2")
#do
#    ((j = "$2"*"$1" + "$i"))
#  sbatch run_simulation.sh "$j"
#done
