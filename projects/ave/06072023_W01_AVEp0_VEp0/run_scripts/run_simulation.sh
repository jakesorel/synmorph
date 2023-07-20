#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=02-00:00:00   # walltime
#SBATCH -J "AVE_simulations"   # job name
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

eval "$(conda shell.bash hook)"
source activate synmorph

python run_multiple_simulation.py ${SLURM_ARRAY_TASK_ID} 1
exit_code=$?

if [ $exit_code -eq 0 ]; then
    # Script completed successfully
    echo "Script completed successfully"
elif [ $exit_code -eq 1 ]; then
    # Restart the script due to TerminatedWorkerError
    echo "Trying without equiangulation..."
    python run_multiple_simulation.py ${SLURM_ARRAY_TASK_ID} 0
    exit_code=$?
else
    # Other non-zero exit code, handle as needed
    echo "Script encountered an error. Exiting..."
fi



#python ../analysis_scripts/run_analysis.py ${SLURM_ARRAY_TASK_ID}