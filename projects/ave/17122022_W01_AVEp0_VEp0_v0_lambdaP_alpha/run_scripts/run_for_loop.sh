#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=3-00:00:00   # walltime
#SBATCH -J "AVE_for_loop_jobs"   # job name
#SBATCH --output=../bash_out/for_loop_jobs_output.out
#SBATCH --error=../bash_out/for_loop_jobs_error.out
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --mem=2G


((Nfull = 551000))
((Nfullm1 = "$Nfull" - 1))
((Nper = "$2"))
((Nperm1 = "$Nper"-1))
((Njob = 551))
((Njobm1 = "$Njob"-1))

#rm ../scan_summary/23032022_W01_AVEp0_VEp0_result_log.txt                                                                                                                         

sbatch --array [0-"$Njobm1"] --nice=200 for_loop.sh "$Nper" "$Nperm1"