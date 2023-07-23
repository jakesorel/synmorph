import numpy as np
import pandas as pd

df = pd.read_csv("../scan_summary/gz_names.txt",header=None)
df["idx"] = [int(nm.split("_")[4]) for nm in df[0]]
is_in = np.zeros(20**4,dtype=int)
for i in df["idx"].values:
    is_in[i] = 1

to_run_idx = np.nonzero(1-is_in)[0]
df_out = pd.DataFrame({"to_run":to_run_idx})
df_out.to_csv("../scan_summary/to_run.csv")
print(df_out.shape)

df_t = pd.read_csv("../analysis_results/t_hit_boundary.txt",header=None)
df_a = pd.read_csv("../scan_summary/analysis_files.txt",header=None)
t_index = [int(nm.split("_")[4]) for nm in df_t[0]]
a_index = [int(nm.split("_")[4]) for nm in df_a[0]]
both_index = list(set(a_index).intersection(set(t_index)))

is_in_a = np.zeros(20**4,dtype=int)
for i in both_index:
    is_in_a[i] = 1

to_run_a_idx = np.nonzero(1-is_in_a)[0]
df_out_a = pd.DataFrame({"to_run":to_run_a_idx})
df_out_a.to_csv("../scan_summary/to_run_analysis.csv")
print(df_out_a.shape)



