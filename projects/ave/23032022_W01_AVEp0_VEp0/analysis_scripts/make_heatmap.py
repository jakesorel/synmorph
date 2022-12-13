import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def make_extent(x_range,y_range,xscale="linear",yscale="linear"):
    if xscale == "log":
        x_range = np.log10(x_range)
    if yscale == "log":
        y_range = np.log10(y_range)
    extent = [x_range[0],x_range[-1],y_range[0],y_range[-1]]
    aspect = (extent[1]-extent[0])/(extent[3]-extent[2])
    return extent,aspect

summary = pd.read_csv("../scan_summary/23032022_W01_AVEp0_VEp0_full_summary.csv")
df = pd.read_csv("../analysis_results/compiled/by_time/299.00.csv")

seed = 2022
seed_space = np.arange(6) + 2022
# AVE_p0 = summary["AVE_p0"].unique()[10]
# W01 = summary["W01"].unique()[-1]
#
# summary_subset = summary[(summary["AVE_p0"] == AVE_p0)*(summary["seed"] == seed)]
# ids = summary_subset["sim_id"]

def get_reduced_summary(summary,ids):
    df_out = pd.DataFrame()
    for id in ids:
        df_out = pd.concat((df_out,summary[summary["sim_index"]==id]))
    return df_out
#
# df_i = get_reduced_summary(df,ids)
# df_j = df_i[["PARAM_VE_p0","PARAM_W01",'AVE_connected_components']]
# df2 = df_j.pivot_table(index=["PARAM_VE_p0"], columns='PARAM_W01')
# # df2.columns
# # g = sns.heatmap(df2)
# # g.set_xticklabels(['{:,.2f}'.format(x)  for x in g.get_xticks()])
# # g.set_yticklabels(['{:,.2f}'.format(x)  for x in g.get_yticks()])
# # plt.show()
# plt.imshow(np.log(df2.values))
# plt.show()

def get_average(df,output,seed_space,cut_axis_name,cut_axis_value,x_axis_name,y_axis_name):
    out = np.zeros((len(seed_space),12,12))
    for i, seed in enumerate(seed_space):
        # summary_subset = summary[(summary["AVE_p0"] == AVE_p0) * (summary["seed"] == seed)]
        summary_subset = summary[(summary[cut_axis_name] == cut_axis_value) * (summary["seed"] == seed)]
        ids = summary_subset["sim_id"]
        df_i = get_reduced_summary(df, ids)
        df_j = df_i[["PARAM_"+y_axis_name, "PARAM_"+x_axis_name, output]]
        df2 = df_j.pivot_table(index=["PARAM_"+y_axis_name], columns="PARAM_"+x_axis_name)
        out[i] = df2.values
    return out.mean(axis=0)
#
#
# plt.plot(df_j["PARAM_VE_p0"].unique(),df2.values[:,5])
# plt.show()
#
# df2.columns = df2.columns.droplevel().rename(None)
#
# fig, ax = plt.subplots()
# y_range = summary["VE_p0"].unique()
# # x_range = summary["W01"].unique()
# x_range = summary["AVE_p0"].unique()
#
# # xlim = (x_range.min(),x_range.max())
# # ylim = (y_range.min(),y_range.max())
# extent,aspect = make_extent(x_range,y_range)
# # av = get_average(df,'AVE_connected_components',seed_space)
# av = get_average(df,'av_av_AVE_d',seed_space)
# cmap = plt.cm.plasma
# ax.imshow(np.flip(av,axis=0),extent=extent,aspect=aspect,cmap=cmap)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=np.max(av), vmin=np.min(av)))
# fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
# cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.06, aspect=18, orientation="vertical")
# cl.set_label("Average AVE displacement")
# ax.set(xlabel="AVE "r"$p_0$",ylabel="VE "r"$p_0$")
# fig.show()
#
#
#
# fig, ax = plt.subplots()
# x_axis_name = "W01"
# y_axis_name = "VE_p0"
# cut_axis_name = "AVE_p0"
# cut_axis_value = summary[cut_axis_name].unique()[-1]
# y_range = summary[y_axis_name].unique()
# x_range = summary[x_axis_name].unique()
# extent,aspect = make_extent(x_range,y_range)
# av_d = get_average(df,'av_av_AVE_d',seed_space,cut_axis_name,cut_axis_value,x_axis_name,y_axis_name)
# cmap = plt.cm.plasma
# ax.imshow(np.flip(av,axis=0),extent=extent,aspect=aspect,cmap=cmap)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=np.max(av), vmin=np.min(av)))
# fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
# cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.06, aspect=18, orientation="vertical")
# cl.set_label("Average AVE displacement")
# ax.set(xlabel="AVE "r"$p_0$",ylabel="VE "r"$p_0$")
# fig.show()

if not os.path.exists("../analysis_plots/"):
    os.mkdir("../analysis_plots/")

if not os.path.exists("../analysis_plots/av_AVE_connectivity"):
    os.mkdir("../analysis_plots/av_AVE_connectivity")

if not os.path.exists("../analysis_plots/av_AVE_connectivity/by_AVE_p0"):
    os.mkdir("../analysis_plots/av_AVE_connectivity/by_AVE_p0")



x_axis_name = "W01"
y_axis_name = "VE_p0"
cut_axis_name = "AVE_p0"
cut_axis_values = summary[cut_axis_name].unique()
y_range = summary[y_axis_name].unique()
x_range = summary[x_axis_name].unique()
for cut_axis_value in cut_axis_values:

    fig, ax = plt.subplots(figsize=(4,4))
    extent,aspect = make_extent(x_range,y_range,xscale="log")
    av = get_average(df,'AVE_connected_components',seed_space,cut_axis_name,cut_axis_value,x_axis_name,y_axis_name)
    cmap = plt.cm.inferno
    vmax=19
    vmin=1
    ax.imshow(np.flip(av,axis=0),extent=extent,aspect=aspect,cmap=cmap,vmax=vmax,vmin=vmin)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=vmax,vmin=vmin))
    fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
    cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.05, aspect=18, orientation="vertical")
    cl.set_label("Average AVE connectivity")
    ax.set(xlabel=r"$log_{10} \ J_{het}$", ylabel="VE "r"$p_0$")
    ax.set_title(cut_axis_name+" = %.2f"%cut_axis_value)
    fig.subplots_adjust(top=0.8,bottom=0.3,left=0.3,right=0.8)
    fig.savefig("../analysis_plots/av_AVE_connectivity/by_AVE_p0/AVE_p0=%.2f.pdf"%cut_axis_value)
    plt.close("all")
if not os.path.exists("../analysis_plots/av_AVE_connectivity/by_VE_p0"):
    os.mkdir("../analysis_plots/av_AVE_connectivity/by_VE_p0")

x_axis_name = "W01"
y_axis_name = "AVE_p0"
cut_axis_name = "VE_p0"
cut_axis_values = summary[cut_axis_name].unique()
y_range = summary[y_axis_name].unique()
x_range = summary[x_axis_name].unique()
for cut_axis_value in cut_axis_values:
    fig, ax = plt.subplots(figsize=(4, 4))
    extent,aspect = make_extent(x_range,y_range,xscale="log")
    av = get_average(df, 'AVE_connected_components', seed_space, cut_axis_name, cut_axis_value, x_axis_name,
                     y_axis_name)
    cmap = plt.cm.inferno
    vmax = 19
    vmin = 1
    ax.imshow(np.flip(av, axis=0), extent=extent, aspect=aspect, cmap=cmap, vmax=vmax, vmin=vmin)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=vmax, vmin=vmin))
    fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
    cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.05, aspect=18, orientation="vertical")
    cl.set_label("Average AVE connectivity")
    ax.set(xlabel=r"$log_{10} \ J_{het}$", ylabel="AVE "r"$p_0$")
    ax.set_title(cut_axis_name + " = %.2f" % cut_axis_value)
    fig.subplots_adjust(top=0.8, bottom=0.3, left=0.3, right=0.8)
    fig.savefig("../analysis_plots/av_AVE_connectivity/by_%s/%s=%.2f.pdf" % (cut_axis_name,cut_axis_name,cut_axis_value))

    plt.close("all")




x_axis_name = "W01"
y_axis_name = "VE_p0"
cut_axis_name = "AVE_p0"
cut_axis_values = summary[cut_axis_name].unique()
y_range = summary[y_axis_name].unique()
x_range = summary[x_axis_name].unique()
if not os.path.exists("../analysis_plots/av_dist/by_%s/"%cut_axis_name):
    os.mkdir("../analysis_plots/av_dist/by_%s/"%cut_axis_name)

for cut_axis_value in cut_axis_values:
    fig, ax = plt.subplots(figsize=(4, 4))
    extent,aspect = make_extent(x_range,y_range,xscale="log")
    av = get_average(df, 'av_av_AVE_d', seed_space, cut_axis_name, cut_axis_value, x_axis_name,
                     y_axis_name)
    cmap = plt.cm.viridis
    vmax = 6
    vmin = 0
    ax.imshow(np.flip(av, axis=0), extent=extent, aspect=aspect, cmap=cmap, vmax=vmax, vmin=vmin)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=vmax, vmin=vmin))
    fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
    cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.05, aspect=18, orientation="vertical")
    cl.set_label("Average AVE displacement")
    ax.set(xlabel=r"$log_{10} \ J_{het}$", ylabel="VE "r"$p_0$")
    ax.set_title(cut_axis_name + " = %.2f" % cut_axis_value)
    fig.subplots_adjust(top=0.8, bottom=0.3, left=0.3, right=0.8)
    fig.savefig("../analysis_plots/av_dist/by_%s/%s=%.2f.pdf" % (cut_axis_name,cut_axis_name,cut_axis_value))

    plt.close("all")


x_axis_name = "W01"
y_axis_name = "AVE_p0"
cut_axis_name = "VE_p0"
cut_axis_values = summary[cut_axis_name].unique()
y_range = summary[y_axis_name].unique()
x_range = summary[x_axis_name].unique()
if not os.path.exists("../analysis_plots/av_dist/by_%s/"%cut_axis_name):
    os.mkdir("../analysis_plots/av_dist/by_%s/"%cut_axis_name)

for cut_axis_value in cut_axis_values:
    fig, ax = plt.subplots(figsize=(4, 4))
    extent,aspect = make_extent(x_range,y_range,xscale="log")
    av = get_average(df, 'av_av_AVE_d', seed_space, cut_axis_name, cut_axis_value, x_axis_name,
                     y_axis_name)
    cmap = plt.cm.inferno
    vmax = 8
    vmin = 0
    ax.imshow(np.flip(av, axis=0), extent=extent, aspect=aspect, cmap=cmap, vmax=vmax, vmin=vmin)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=vmax, vmin=vmin))
    fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
    cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.05, aspect=18, orientation="vertical")
    cl.set_label("Average AVE displacement")
    ax.set(xlabel=r"$log_{10} \ J_{het}$", ylabel="AVE "r"$p_0$")
    ax.set_title(cut_axis_name + " = %.2f" % cut_axis_value)
    fig.subplots_adjust(top=0.8, bottom=0.3, left=0.3, right=0.8)
    fig.savefig("../analysis_plots/av_dist/by_%s/%s=%.2f.pdf" % (cut_axis_name,cut_axis_name,cut_axis_value))

    plt.close("all")



x_axis_name = "AVE_p0"
y_axis_name = "VE_p0"
cut_axis_name = "W01"
cut_axis_values = summary[cut_axis_name].unique()
y_range = summary[y_axis_name].unique()
x_range = summary[x_axis_name].unique()
if not os.path.exists("../analysis_plots/av_dist/by_%s/"%cut_axis_name):
    os.mkdir("../analysis_plots/av_dist/by_%s/"%cut_axis_name)

for cut_axis_value in cut_axis_values:
    fig, ax = plt.subplots(figsize=(4, 4))
    extent,aspect = make_extent(x_range,y_range)
    av = get_average(df, 'av_av_AVE_d', seed_space, cut_axis_name, cut_axis_value, x_axis_name,
                     y_axis_name)
    cmap = plt.cm.viridis
    vmax = 6
    vmin = 0
    ax.imshow(np.flip(av, axis=0), extent=extent, aspect=aspect, cmap=cmap, vmax=vmax, vmin=vmin)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=vmax, vmin=vmin))
    fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
    cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.05, aspect=18, orientation="vertical")
    cl.set_label("Average AVE displacement")
    ax.set(xlabel="AVE "r"$p_0$", ylabel="VE "r"$p_0$")
    ax.set_title(r"$log_{10}$" + " " + cut_axis_name + " = %.2f" % np.log10(cut_axis_value))
    fig.subplots_adjust(top=0.8, bottom=0.3, left=0.3, right=0.8)
    fig.savefig("../analysis_plots/av_dist/by_%s/log%s=%.2f.pdf" % (cut_axis_name,cut_axis_name,np.log10(cut_axis_value)))

    plt.close("all")

