#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=23:59:00   # walltime
#SBATCH -J "clear dump"   # job name
#SBATCH --output=../bash_out/outputclear_dump.out
#SBATCH --error=../bash_out/error_clear_dump.out
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G



rsync -a --delete _dump/ dump/
