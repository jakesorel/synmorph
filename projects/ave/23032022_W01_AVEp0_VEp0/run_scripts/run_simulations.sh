#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=5:00:00   # walltime
#SBATCH -J "AVE_simulations"   # job name
#SBATCH --output=../bash_out/output.out
#SBATCH --error=../bash_out/error.out
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --mem=2G


eval "$(conda shell.bash hook)"
source activate synmorph


for i in $(seq 0 "$2")
do
    ((j = "$2"*"$1" + "$i"))
  python run_single_simulation.py "$j"
done
