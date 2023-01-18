#!/bin/bash 

#Submit this script with: sbatch thefilename                                                                                                                                       

#SBATCH --time=00:20:00   # walltime                                                                                                                                             
#SBATCH -J "AVE2_for_loop_jobs"   # job name                                                                                                                                      
#SBATCH --output=../bash_out/for_loop_2nd_jobs_output.out                                                                                                                              
#SBATCH --error=../bash_out/for_loop_2nd_jobs_error.out                                                                                                                                
#SBATCH -n 1                                                                                                                                                                       
#SBATCH --partition=cpu                                                                                                                                                            
#SBATCH --mem=2G                                                                                                                                                                   





sbatch --array [0-"$2"] run_simulation.sh ${SLURM_ARRAY_TASK_ID} "$1" 

