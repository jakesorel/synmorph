import os
import sys


SCRIPT_DIR = "../../../../"
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import pandas as pd
from joblib import Parallel,delayed

def extract_scores(i):
    try:
        df = pd.read_csv('../analysis_results/all/02102023_AVEp0_VEp0_%i_analysis.csv'%i)
        d = df["av_av_AVE_d"].values
        max_AVE_distance = d.max()
        percentile95_distance = np.percentile(d,95)
        mask95 = d>percentile95_distance
        t_95_distance,AVE_contiguous = np.nan,np.nan
        if mask95.size!=0:
            t_95_distance = df["t"].values[mask95][0]
            AVE_contiguous = df["AVE_connected_components"].values[mask95][0]
        out = i,max_AVE_distance,percentile95_distance,t_95_distance,AVE_contiguous
    except:
        out = i, np.nan,np.nan,np.nan,np.nan
    file = open("../analysis_results/minimal/%i.txt"%i,"w+")
    file.write(",".join(np.array(out).astype(str)) + "\n")
    file.close()

if __name__ == "__main__":

    if not os.path.exists("../analysis_results/minimal"):
        os.mkdir("../analysis_results/minimal")
    N = 20
    M = 20
    total_sims = N ** 2 * M
    sims_per_lot = 20
    slurm_index = int(sys.argv[1])
    print("Slurm index", slurm_index)
    range_to_sample = np.arange(slurm_index * sims_per_lot, (slurm_index + 1) * sims_per_lot)

    results = np.array(Parallel(n_jobs=-1, backend="loky", prefer="threads")(delayed(extract_scores)(i) for i in range_to_sample))
