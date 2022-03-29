import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

summary = pd.read_csv("../scan_summary/23032022_W01_AVEp0_VEp0_full_summary.csv")
df = pd.read_csv("../analysis_results/compiled/by_time/299.00.csv")

seed = 2023
AVE_p0 = summary["VE_p0"].unique()[-1]

summary_subset = summary[(summary["AVE_p0"] == AVE_p0)*(summary["seed"] == seed)]
ids = summary_subset["sim_id"]

def get_reduced_summary(summary,ids):
    df_out = pd.DataFrame()
    for id in ids:
        df_out = pd.concat((df_out,summary[summary["sim_index"]==id]))
    return df_out

df_i = get_reduced_summary(df,ids)
df_j = df_i[["PARAM_VE_p0","PARAM_W01","av_av_AVE_d"]]
df2 = df_j.pivot_table(index=["PARAM_VE_p0"], columns='PARAM_W01')
# df2.columns
# g = sns.heatmap(df2)
# g.set_xticklabels(['{:,.2f}'.format(x)  for x in g.get_xticks()])
# g.set_yticklabels(['{:,.2f}'.format(x)  for x in g.get_yticks()])
# plt.show()
plt.imshow(df2.values)
plt.show()

df2.columns = df2.columns.droplevel().rename(None)