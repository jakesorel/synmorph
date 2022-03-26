#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=04:00:00   # walltime
#SBATCH -J "AVE_analysis"   # job name
#SBATCH --output=../bash_out/output_analysis.out
#SBATCH --error=../bash_out/error_analysis.out
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --mem=2G

((j = "$2"*"$1"))
((terminate = "$2"*"$1" + "$2"))
eval "$(conda shell.bash hook)"
source activate synmorph

for i in $(seq 0 "$2")
do
    ((j = "$2"*"$1" + "$i"))
  python ../analysis_scripts/run_analysis.py "$1"
done
