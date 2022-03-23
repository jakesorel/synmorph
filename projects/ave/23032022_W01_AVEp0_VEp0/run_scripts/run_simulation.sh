#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=0:15:00   # walltime
#SBATCH -J "AVE_simulations"   # job name
#SBATCH --output=../bash_out/output.out
#SBATCH --error=../bash_out/error.out
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --mem=2G


eval "$(conda shell.bash hook)"
source activate synmorph
python run_single_simulation.py "$1"


((jp1 = "$1" + 1))

if [[ "$1" -lt "$2" ]]
then
    sbatch run_simulation.sh "$jp1" "$2"
fi

