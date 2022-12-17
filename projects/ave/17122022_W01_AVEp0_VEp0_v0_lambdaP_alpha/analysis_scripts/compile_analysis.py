"""
Combined analysis files will be too large.

So compile the relevant results by time.
"""
import os
import pandas as pd
import numpy as np

if not os.path.exists("../analysis_results/compiled"):
    os.mkdir("../analysis_results/compiled")

if not os.path.exists("../analysis_results/compiled/by_time"):
    os.mkdir("../analysis_results/compiled/by_time")

file_names = os.listdir("../analysis_results/all")
sim_id = [int(fl.split("_VEp0_")[1].split("_analysis")[0]) for fl in file_names]
csv_files = ["../analysis_results/all/" + fl for fl in file_names]
column_names = list(pd.read_csv(csv_files[0]).columns.drop("t"))
t_span = pd.read_csv(csv_files[0])["t"].values

t_dfs = []
for i in range(len(t_span)):
    t_dfs += [np.zeros((len(csv_files),len(column_names)))]


for i, csv_file in enumerate(csv_files):
    df = pd.read_csv(csv_file)
    contents = df.values[:,1:]
    for ti in range(len(t_span)):
        t_dfs[ti][i] = contents[ti]

for i, t in enumerate(t_span):
    dictionary = {"sim_index":sim_id}
    dictionary.update(zip(column_names,t_dfs[i].T))
    df = pd.DataFrame(dictionary)
    df["sim_index"] = sim_id
    df.to_csv("../analysis_results/compiled/by_time/%.2f.csv"%t,index=None)

