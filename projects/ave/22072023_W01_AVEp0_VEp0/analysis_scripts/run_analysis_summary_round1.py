import os
import sys


SCRIPT_DIR = "../../../../"
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import pandas as pd
from joblib import Parallel,delayed

def extract_scores(i):
    try:
        df = pd.read_csv('../analysis_results/all/22072023_W01_AVEp0_VEp0_%i_analysis.csv'%i)
        d = df["av_av_AVE_d"].values
        max_AVE_distance = d.max()
        percentile95_distance = np.percentile(d,95)
        mask95 = d>percentile95_distance
        t_95_distance,AVE_contiguous = np.nan,np.nan
        if mask95.size!=0:
            t_95_distance = df["t"].values[mask95][0]
            AVE_contiguous = df["AVE_connected_components"].values[mask95][0]
        return max_AVE_distance,percentile95_distance,t_95_distance,AVE_contiguous
    except:
        return np.nan,np.nan,np.nan,np.nan

if __name__ == "__main__":

    if not os.path.exists("../analysis_results/minimal"):
        os.mkdir("../analysis_results/minimal")
    N = 20
    total_sims = N**4
    sims_per_lot = 400
    slurm_index = int(sys.argv[1])
    range_to_sample = np.arange(slurm_index*sims_per_lot,(slurm_index+1)*sims_per_lot)

    results = np.array(Parallel(n_jobs=-1, backend="loky", prefer="threads")(delayed(extract_scores)(i) for i in range_to_sample))
    df_out = pd.DataFrame(results)
    df_out.columns = "max_AVE_distance,percentile95_distance,t_95_distance,AVE_contiguous".split(",")
    df_out.index = range_to_sample
    df_out.to_csv("../analysis_results/minimal/block_%d.csv"%slurm_index)