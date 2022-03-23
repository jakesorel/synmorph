#!/bin/bash

((Nfull = "$1"*"$1"*"$1"*"$2"))
((Nper = "$1"*"$2"))
((Njob = "$1"*"$1"))


for i in $(seq 0 $(($Njob-1)))
do
#    ./run_simulations.sh "$i" "$Nper"
    sbatch run_simulations.sh "$i" "$Nper"
done