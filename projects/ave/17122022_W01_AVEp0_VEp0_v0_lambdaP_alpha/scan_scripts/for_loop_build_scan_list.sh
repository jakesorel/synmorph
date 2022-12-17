#!/bin/bash
#Submit this script with: sbatch thefilename

#SBATCH --time=2:00:00   # walltime
#SBATCH -J "AVE_for_loop_build_lists"   # job name
#SBATCH --output=../bash_out/for_loop_build_lists_output.out
#SBATCH --error=../bash_out/for_loop_build_lists_error.out
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --mem=2G

((Nfull = "$1"*"$1"*"$1"*"$1"*"$1"*"$1"*"$2"))
((Nfullm1 = "$Nfull" - 1))
((Nper = "$1"*"$2"))
((Nperm1 = "$1"*"$2"-1))
((Njob = "$1"*"$1"*"$1"*"$1"*"$1"))
((Njobm1 = "$1"*"$1"*"$1"*"$1"*"$1"-1))

rm -R ../scan_summary
#rm -R ../scan_dicts

mkdir scan_summary
#mkdir scan_dicts

for i in $(seq 0 "$Njobm1")
do
  sbatch --array [0-"$Nperm1"] build_scan_list.sh "$i" "$Nper"
done
