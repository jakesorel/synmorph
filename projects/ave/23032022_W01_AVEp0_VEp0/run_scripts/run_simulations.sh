#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=5:00:00   # walltime
#SBATCH -J "AVE_simulation_loader"   # job name
#SBATCH --output=../bash_out/output_loader.out
#SBATCH --error=../bash_out/error_loader.out
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --mem=50M


for i in $(seq 0 "$2")
do
    ((j = "$2"*"$1" + "$i"))
  sbatch run_simulation.sh "$j"
done
