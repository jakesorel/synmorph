#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=00-01:00:00   # walltime
#SBATCH -J "AVE_simulations"   # job name
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G

eval "$(conda shell.bash hook)"
source activate synmorph

while true; do
    python run_multiple_simulation.py ${SLURM_ARRAY_TASK_ID}
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        # Script completed successfully
        echo "Script completed successfully"
        break
    elif [ $exit_code -eq 1 ]; then
        # Restart the script due to TerminatedWorkerError
        echo "Restarting script..."
    else
        # Other non-zero exit code, handle as needed
        echo "Script encountered an error. Exiting..."
        break
    fi
done


#python ../analysis_scripts/run_analysis.py ${SLURM_ARRAY_TASK_ID}