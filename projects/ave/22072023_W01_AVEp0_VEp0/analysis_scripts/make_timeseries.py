import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


summary = pd.read_csv("../scan_summary/23032022_W01_AVEp0_VEp0_full_summary.csv")
df = pd.read_csv("../analysis_results/compiled/by_time/299.00.csv")
by_time_dir = "../analysis_results/compiled/by_time/"

seed = 2022
seed_space = np.arange(6) + 2022

if not os.path.exists("../analysis_plots/time_series"):
    os.mkdir("../analysis_plots/time_series")

if not os.path.exists("../analysis_plots/time_series/misc"):
    os.mkdir("../analysis_plots/time_series/misc")


def get_reduced_summary(summary,ids):
    df_out = pd.DataFrame()
    for id in ids:
        df_out = pd.concat((df_out,summary[summary["sim_index"]==id]))
    return df_out

def get_average_time_val(df,output,seed_space,cut_axis_name1,cut_axis_value1,cut_axis_name2,cut_axis_value2):
    out = np.zeros((len(seed_space),12))
    for i, seed in enumerate(seed_space):
        # summary_subset = summary[(summary["AVE_p0"] == AVE_p0) * (summary["seed"] == seed)]
        summary_subset = summary[(summary[cut_axis_name1] == cut_axis_value1) * (summary[cut_axis_name2] == cut_axis_value2) * (summary["seed"] == seed)]
        ids = summary_subset["sim_id"]
        df_i = get_reduced_summary(df, ids)
        out[i] = df_i[output].values
    return out.mean(axis=0)

t_span = np.arange(0.5,300,5)
t_span_strings = ["%.2f"%t for t in t_span]

def get_average_time_series(t_span_strings,by_time_dir,output,seed_space,cut_axis_name1,cut_axis_value1,cut_axis_name2,cut_axis_value2):
    out = np.ones((len(t_span_strings),12))*np.nan
    for i in range(len(t_span_strings)):
        t = t_span_strings[i]
        file_name = by_time_dir + t + ".csv"
        try:
            df = pd.read_csv(file_name)
            out[i] = get_average_time_val(df,output,seed_space,cut_axis_name1,cut_axis_value1,cut_axis_name2,cut_axis_value2)
        except:
            out[i] = np.nan
    return out

if not os.path.exists("../analysis_plots/time_series/misc/by_AVE_p0"):
    os.mkdir("../analysis_plots/time_series/misc/by_AVE_p0")


cut_axis_name1 = "AVE_p0"
cut_axis_value1s = summary[cut_axis_name1].unique()
cut_axis_name2 = "W01"
cut_axis_value2 = summary[cut_axis_name2].unique()[-2]
x_name = "VE_p0"
# output = "bulk_AVE_speed"
# output = 'AVE_connected_components'
output = "av_av_AVE_d"
for cut_axis_value1 in cut_axis_value1s:
    time_series = get_average_time_series(t_span_strings,by_time_dir,output,seed_space,cut_axis_name1,cut_axis_value1,cut_axis_name2,cut_axis_value2)
    keep = ~np.any(np.isnan(time_series),axis=1)

    x_range = summary["VE_p0"].unique()
    cmap = plt.cm.plasma
    cols = cmap(np.linspace(0,1,12))
    fig, ax = plt.subplots(figsize=(4,4))
    for i in range(12):
        ax.plot(t_span[keep],time_series[keep,i],color=cols[i])
    ax.set_title("AVE p0 = %.2f"%cut_axis_value1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=x_range.max(), vmin=x_range.min()))
    cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.05, aspect=18, orientation="vertical")
    cl.set_label("VE "r"$p_0$")
    ax.set(xlabel="Time",ylabel="Average AVE displacement")
    ax.set(ylim=(0,8))
    fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
    fig.savefig("../analysis_plots/time_series/misc/by_AVE_p0/av_displacement by VE p0 AVEp0%.2f.pdf"%cut_axis_value1,dpi=300)

output = "bulk_AVE_speed"
for cut_axis_value1 in cut_axis_value1s:
    time_series = get_average_time_series(t_span_strings,by_time_dir,output,seed_space,cut_axis_name1,cut_axis_value1,cut_axis_name2,cut_axis_value2)
    keep = ~np.any(np.isnan(time_series),axis=1)

    x_range = summary["VE_p0"].unique()
    cmap = plt.cm.plasma
    cols = cmap(np.linspace(0,1,12))
    fig, ax = plt.subplots(figsize=(4,4))
    for i in range(12):
        ax.plot(t_span[keep],time_series[keep,i],color=cols[i])
    ax.set_title("AVE p0 = %.2f"%cut_axis_value1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=x_range.max(), vmin=x_range.min()))
    cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.05, aspect=18, orientation="vertical")
    cl.set_label("VE "r"$p_0$")
    ax.set(xlabel="Time",ylabel="Average AVE speed")
    # ax.set(ylim=(0,8))
    fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
    fig.savefig("../analysis_plots/time_series/misc/by_AVE_p0/av_speed by VE p0 AVEp0%.2f.pdf"%cut_axis_value1,dpi=300)

