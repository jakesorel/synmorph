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


