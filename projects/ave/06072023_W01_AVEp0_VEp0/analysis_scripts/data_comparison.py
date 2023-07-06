import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import pickle

"""
Questions: how are quadrants assigned. 

How do you get cumulative. 
"""

###In the real dataset (1,0) is the direction of the AVE.

df_embryo = pd.read_csv("../reference/L455_E3.csv")
df_embryo = df_embryo[(df_embryo["Static_Quad_ID"]>0)*(df_embryo["Static_Quad_ID"]<17)]
df_embryo["ecc"] = np.sqrt(np.square(df_embryo['Major_Length']/2) - np.square(df_embryo['Minor_Length']/2)) / (df_embryo['Major_Length']/2)
df_embryo["N_neighbours"] = [len(trk.split(":")) for trk in df_embryo["Cell_Neighbours_Track_ID"]]
# df_embryo = df_embryo[df_embryo["Stage"] == 'Migration to Boundary']

df_embryo_by_quad_id = pd.DataFrame()

frames = np.unique(df_embryo["Frame"])[:-1]
quadrants = np.unique(df_embryo["Static_Quad_ID"])

for frame in frames:
    dfi = df_embryo[df_embryo["Frame"]==frame]
    df_by_q_i = dfi.groupby("Static_Quad_ID").agg(np.nanmean)
    # print(dfi.groupby("Static_Quad_ID").count()["row_id"])
    # df_by_q_i_sum = dfi.groupby("Static_Quad_ID").agg(np.nansum)
    df_by_q_i["Static_Quad_ID"] = df_by_q_i.index
    df_embryo_by_quad_id = pd.concat([df_embryo_by_quad_id,df_by_q_i])

columns_embryo = ["Area",
                  "Perimeter",
                  "Shape_Index",
                  "ecc",
                  "N_neighbours",
                  "speed_3D",
                  "AVE_direction_speed_3D"]

columns_simulation = ["A",
                      "P",
                      "SI",
                      "ecc",
                      "N_neighbours",
                      "inst_speed",
                      "inst_ant_speed"]

embryo_statistics_by_quadrant = np.zeros((len(frames),len(columns_embryo),len(quadrants)))
for i, frame in enumerate(frames):
    dfi = df_embryo_by_quad_id[(df_embryo_by_quad_id["Frame"]==frame)]
    embryo_statistics_by_quadrant[i] = [dfi[c].values for c in columns_embryo]


z_embryo_statistics_by_quadrant = embryo_statistics_by_quadrant.copy()
z_embryo_statistics_by_quadrant = z_embryo_statistics_by_quadrant - np.expand_dims(np.nanmean(embryo_statistics_by_quadrant,axis=(0,2)),axis=(0,2))
z_embryo_statistics_by_quadrant = z_embryo_statistics_by_quadrant/np.expand_dims(np.nanstd(embryo_statistics_by_quadrant,axis=(0,2)),axis=(0,2))

file_names = os.listdir("../analysis_results/compiled/by_position_class")
if ".DS_Store" in file_names:
    file_names.remove(".DS_Store")
sim_ids = [int(fl.split("_VEp0_")[1].split("_analysis")[0]) for fl in file_names]
sim_csv_files = ["../analysis_results/compiled/by_position_class/" + fl for fl in file_names]

df_sim = pd.read_csv(sim_csv_files[0])
column_ids = [list(df_sim.columns).index(c) for c in columns_simulation]
frames_sim = np.unique(df_sim["t"])
q=0
simulation_statistics_by_quadrant_by_simulation = np.zeros((len(sim_ids),len(frames_sim),16,len(column_ids)))
for i, (sim_id, sim_csv_file) in enumerate(zip(sim_ids,sim_csv_files)):
    print(q)
    df_sim = pd.read_csv(sim_csv_file)
    simulation_statistics_by_quadrant = df_sim.values.T[column_ids].T.reshape(len(frames_sim),16,len(column_ids))
    simulation_statistics_by_quadrant_by_simulation[i] = simulation_statistics_by_quadrant
    q+=1

W01,AVE_p0,nonAVE_p0,random_seed = np.zeros(len(sim_ids)),np.zeros(len(sim_ids)),np.zeros(len(sim_ids)),np.zeros(len(sim_ids))
for i, sim_id in enumerate(sim_ids):
    pikd = open("../scan_dicts/23032022_W01_AVEp0_VEp0_%s.pickle" % sim_id, 'rb')
    scan_dict = pickle.load(pikd)
    pikd.close()
    W01[i] = scan_dict["tissue_params"]["W"][0,1]
    AVE_p0[i] = scan_dict["grn_params"]["AVE_p0"]
    nonAVE_p0[i] = scan_dict["grn_params"]["nonAVE_p0"]
    random_seed[i] = scan_dict["simulation_params"]["random_seed"]
    print(i)


simulation_statistics_by_quadrant_by_simulation = np.array(simulation_statistics_by_quadrant_by_simulation)
simulation_statistics_by_quadrant_by_simulation = simulation_statistics_by_quadrant_by_simulation.transpose(0,1,3,2)

#
# z_simulation_statistics_by_quadrant_by_simulation = simulation_statistics_by_quadrant_by_simulation.copy()
# z_simulation_statistics_by_quadrant_by_simulation = z_simulation_statistics_by_quadrant_by_simulation - np.expand_dims(np.nanmean(simulation_statistics_by_quadrant_by_simulation,axis=(1,3)),axis=(1,3))
# z_simulation_statistics_by_quadrant_by_simulation = z_simulation_statistics_by_quadrant_by_simulation / np.expand_dims(np.nanstd(simulation_statistics_by_quadrant_by_simulation,axis=(1,3)),axis=(1,3))
#


z_simulation_statistics_by_quadrant_by_simulation = simulation_statistics_by_quadrant_by_simulation.copy()
z_simulation_statistics_by_quadrant_by_simulation = z_simulation_statistics_by_quadrant_by_simulation - np.expand_dims(np.nanmean(simulation_statistics_by_quadrant_by_simulation,axis=(0,1,3)),axis=(0,1,3))
z_simulation_statistics_by_quadrant_by_simulation = z_simulation_statistics_by_quadrant_by_simulation / np.expand_dims(np.nanstd(simulation_statistics_by_quadrant_by_simulation,axis=(0,1,3)),axis=(0,1,3))


# z_simulation_statistics_by_quadrant_by_simulation = z_simulation_statistics_by_quadrant_by_simulation.transpose(0,1,3,2)

cost_by_sim = np.sum((z_simulation_statistics_by_quadrant_by_simulation[:,-77:] - np.expand_dims(z_embryo_statistics_by_quadrant,0))**2,axis=(1,2,3))

# cost_by_sim = np.nansum((simulation_statistics_by_quadrant_by_simulation/np.expand_dims(np.nanmean(simulation_statistics_by_quadrant_by_simulation,axis=(1,3)),axis=(1,3)) - np.expand_dims(embryo_statistics_by_quadrant[:-1]/np.expand_dims(np.nanmean(embryo_statistics_by_quadrant[:-1],axis=(0,2)),axis=(0,2)),0))**2,axis=(1,2,3))

# cost_by_sim = np.nansum((z_simulation_statistics_by_quadrant_by_simulation[:,:,-2:] - np.expand_dims(z_embryo_statistics_by_quadrant[:-1,-2:],0))**2,axis=(1,2,3))


average_anterior_speed = simulation_statistics_by_quadrant_by_simulation[:,-1,-1]

plt.scatter(average_anterior_speed[:,0],cost_by_sim)
plt.show()

cmap = plt.cm.inferno
fig,ax = plt.subplots(7,4)
j = np.nanargmin(cost_by_sim)
for i in range(7):
    sm = simulation_statistics_by_quadrant_by_simulation[j,:,i]
    sm1 = sm[:,:8]
    sm2 = sm[:,8:]
    ax[i,0].imshow(sm[:,:8],aspect="auto",cmap=cmap)
    ax[i,2].imshow(sm[:,8:],aspect="auto",cmap=cmap)


    ax[i,1].imshow(embryo_statistics_by_quadrant[:,i,:8],aspect="auto",cmap=cmap,interpolation=None)
    ax[i,3].imshow(embryo_statistics_by_quadrant[:,i,8:],aspect="auto",cmap=cmap,interpolation=None)

    ax[i,0].set(ylabel=columns_simulation[i])
fig.show()

titles = ["Sim\ninner8","Data\ninner8","Sim\nouter8","Data\nouter8"]
interp = "bicubic"
cmap = plt.cm.inferno
fig,ax = plt.subplots(7,4,figsize=(8,14),sharex=True,sharey=True)
j = np.nanargmin(cost_by_sim)
for i in range(7):
    sm = simulation_statistics_by_quadrant_by_simulation[j,:,i]
    sm1 = sm[:,:8]
    sm2 = sm[:,8:]
    ax[i,0].imshow(sm1,aspect="auto",cmap=cmap,interpolation=interp,vmin=np.percentile(sm,5),vmax=np.percentile(sm,95))
    ax[i,2].imshow(sm2,aspect="auto",cmap=cmap,interpolation=interp,vmin=np.percentile(sm,5),vmax=np.percentile(sm,95))

    em = embryo_statistics_by_quadrant[:,i]
    em1 = embryo_statistics_by_quadrant[:,i,:8]
    em2 = embryo_statistics_by_quadrant[:,i,8:]

    ax[i,1].imshow(em1,interpolation=interp,aspect="auto",cmap=cmap,vmin=np.nanpercentile(em,5),vmax=np.nanpercentile(em,95))
    ax[i,3].imshow(em2,interpolation=interp,aspect="auto",cmap=cmap,vmin=np.nanpercentile(em,5),vmax=np.nanpercentile(em,95))

    ax[i,0].set(ylabel=columns_simulation[i].replace("_","\n"))
    if i == 0:
        for k, lab in enumerate(titles):
            ax[i, k].set_title(lab)
fig.show()
fig.savefig("time_courses_smoothed.pdf",dpi=300)



def make_polar_plot(ax,values,cmap=None,vmin=None,vmax=None):
    if cmap is None:
        cmap = plt.cm.plasma
    if vmax is None:
        vmax = values.max()
    if vmin is None:
        vmin = values.min()
    values_norm = (values - vmin)/(vmax - vmin)

    explode = np.zeros(8)
    explode[0] = 0.2
    explode[-1] = 0.2
    ax.pie(np.ones(8), explode=explode,labels=None, colors=cmap(values_norm[8:16]),radius=2)
    ax.pie(np.ones(8), explode=explode,labels=None, colors=cmap(values_norm[1:8]),radius=1)


fig,ax = plt.subplots(2,7,figsize=(8,3))
for i in range(7):
    sm = z_simulation_statistics_by_quadrant_by_simulation[j,:,i]
    ti = 100-77+50
    sim = sm[ti]
    emb = z_embryo_statistics_by_quadrant[50,i]
    vmin = np.min(z_simulation_statistics_by_quadrant_by_simulation[j,:,i])
    vmax = np.max(z_embryo_statistics_by_quadrant[:,i])
    make_polar_plot(ax[0,i],sim)#,vmin=vmin,vmax=vmax)
    make_polar_plot(ax[1,i],emb)#,vmin=vmin,vmax=vmax)
    ax[0,i].set_title(columns_simulation[i].replace("_","\n"))

for axx in ax.ravel():
    axx.set(xlim=(-2,2),ylim=(-2,2),aspect=1)
    axx.axis("off")
fig.subplots_adjust(wspace=0.5,hspace=0)
fig.show()
fig.savefig("statistics_comparison.pdf")




W01_range = np.unique(W01)
AVEp0_range = np.unique(AVE_p0)
VEp0_range = np.unique(nonAVE_p0)

cost_sim_square = cost_by_sim[np.argsort(sim_ids)].reshape(12,12,12,6)
cost_sim_square_average = cost_sim_square.mean(axis=-1)

W01_square = W01[np.argsort(sim_ids)].reshape(12,12,12,6).mean(axis=-1)
VEp0_square = nonAVE_p0[np.argsort(sim_ids)].reshape(12,12,12,6).mean(axis=-1)

c = stats.rankdata(cost_sim_square_average.ravel(),"average").reshape(cost_sim_square_average.shape)/cost_sim_square_average.size

# c = (np.argsort(cost_sim_square_average.ravel())/len(cost_sim_square_average.ravel())).reshape(cost_sim_square_average.shape)

for i in range(12):
    fig, ax = plt.subplots()
    ax.imshow(c[:,:,i],vmin=0,vmax=1,interpolation=None)
    fig.savefig("vep0%d.pdf"%i,dpi=300)
plt.close("all")



for i in range(12):
    fig, ax = plt.subplots()
    ax.imshow(W01_square[:,:,i],interpolation=None)
    fig.savefig("vep0%d_w01.pdf"%i,dpi=300)
plt.close("all")


for i in range(12):
    fig, ax = plt.subplots()
    ax.imshow(-c[:,i],vmin=-1,vmax=0,interpolation=None)
    fig.savefig("avep0%d.pdf"%i,dpi=300)
plt.close("all")


df_fit = pd.DataFrame({"random_seed":random_seed,"AVE_p0":AVE_p0,"nonAVE_p0":nonAVE_p0,"W01":W01,"cost":cost_by_sim})




# #
# #
# dfi = df_embryo.groupby("Static_Quad_ID").mean()
# #
# plt.scatter(df_embryo["pos_2D_x"],df_embryo["pos_2D_y"],c=df_embryo["Static_Quad_ID"],alpha=1)
# plt.show()
#
# dfi = df_sim.groupby("position_class").mean()
# #
# plt.scatter(df_sim["x"],df_sim["y"],c=df_sim["position_class"],alpha=1)
# plt.show()


"""
I think the comparisons are not 'close' enough. 


"""

fig, ax = plt.subplots(7)
for i in range(7):
    ax[i].hist(embryo_statistics_by_quadrant[:,i].ravel(),bins=100)
fig.show()


fig, ax = plt.subplots(7,figsize=(4,10))
for i in range(7):
    for q in range(8):
        ax[i].plot(embryo_statistics_by_quadrant[:,i,q])
fig.show()

fig, ax = plt.subplots()
i = 2
for q in range(8):
    ax.plot(embryo_statistics_by_quadrant[:,i,q])
ax.set(ylim=(3,6))
fig.show()


z_embryo_statistics_by_quadrant_time_binned = np.zeros(((int(len(embryo_statistics_by_quadrant)/10),)+embryo_statistics_by_quadrant.shape[1:]))
for i in range(int(len(embryo_statistics_by_quadrant)/10)):
    z_embryo_statistics_by_quadrant_time_binned[i] = np.nanmean(z_embryo_statistics_by_quadrant[i*10:(i+1)*10],axis=0)

z_simulation_statistics_by_quadrant_by_simulation_time_binned = np.zeros(((simulation_statistics_by_quadrant_by_simulation.shape[0],)+(int(simulation_statistics_by_quadrant_by_simulation.shape[1]/10),)+simulation_statistics_by_quadrant_by_simulation.shape[2:]))
for i in range(int(simulation_statistics_by_quadrant_by_simulation.shape[1]/10)):
    z_simulation_statistics_by_quadrant_by_simulation_time_binned[:,i] = np.nanmean(z_simulation_statistics_by_quadrant_by_simulation[:,i*10:(i+1)*10],axis=1)

fig, ax = plt.subplots(7,figsize=(4,10))
for i in range(7):
    for q in range(8):
        ax[i].plot(z_embryo_statistics_by_quadrant_time_binned[:,i,q],label=q)
        # ax[i].legend()
fig.show()


fig, ax = plt.subplots(7,figsize=(4,10))
for i in range(7):
    for q in range(8):
        ax[i].plot(z_simulation_statistics_by_quadrant_by_simulation_time_binned[j, :,i,q],label=q)
        # ax[i].legend()
fig.show()




cost_by_sim = np.nansum(np.sqrt((z_simulation_statistics_by_quadrant_by_simulation_time_binned - np.expand_dims(z_embryo_statistics_by_quadrant_time_binned,0))**2),axis=(1,2,3))



titles = ["Sim\ninner8","Data\ninner8","Sim\nouter8","Data\nouter8"]
interp = "bicubic"
cmap = plt.cm.inferno
fig,ax = plt.subplots(7,4,figsize=(8,14),sharex=True,sharey=True)
# j = np.argmin(cost_by_sim)
for i in range(7):
    sm = z_simulation_statistics_by_quadrant_by_simulation_time_binned[j,:,i]
    sm1 = sm[:,:8]
    sm2 = sm[:,8:]
    ax[i,0].imshow(sm1,aspect="auto",cmap=cmap,interpolation=interp,vmin=np.percentile(sm,5),vmax=np.percentile(sm,95))
    ax[i,2].imshow(sm2,aspect="auto",cmap=cmap,interpolation=interp,vmin=np.percentile(sm,5),vmax=np.percentile(sm,95))

    em = z_embryo_statistics_by_quadrant_time_binned[:,i]
    em1 = z_embryo_statistics_by_quadrant_time_binned[:,i,:8]
    em2 = z_embryo_statistics_by_quadrant_time_binned[:,i,8:]

    ax[i,1].imshow(em1,interpolation=interp,aspect="auto",cmap=cmap,vmin=np.nanpercentile(em,5),vmax=np.nanpercentile(em,95))
    ax[i,3].imshow(em2,interpolation=interp,aspect="auto",cmap=cmap,vmin=np.nanpercentile(em,5),vmax=np.nanpercentile(em,95))

    ax[i,0].set(ylabel=columns_simulation[i].replace("_","\n"))
    if i == 0:
        for k, lab in enumerate(titles):
            ax[i, k].set_title(lab)
fig.show()


cost_by_sim


df = pd.read_csv("../analysis_results/compiled/by_time/299.00.csv")

summary = pd.read_csv("../scan_summary/23032022_W01_AVEp0_VEp0_full_summary.csv")

def get_reduced_summary(summary,ids):
    df_out = pd.DataFrame()
    for id in ids:
        df_out = pd.concat((df_out,summary[summary["sim_index"]==id]))
    return df_out
#
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

x_axis_name = "AVE_p0"
y_axis_name = "VE_p0"
cut_axis_name = "W01"
cut_axis_values = summary[cut_axis_name].unique()
y_range = summary[y_axis_name].unique()
x_range = summary[x_axis_name].unique()

seed = 2022
seed_space = np.arange(6) + 2022

av = get_average(df,'av_av_AVE_d',seed_space,cut_axis_name,cut_axis_values[-2],x_axis_name,y_axis_name)

j = 1223

j = np.argmin(cost_by_sim)


z_sim_sample = z_simulation_statistics_by_quadrant_by_simulation_time_binned[j]
z_emb = z_embryo_statistics_by_quadrant_time_binned.copy()

fig, ax = plt.subplots(4,2)
ax = ax.ravel()
for i in range(7):
    ax[i].imshow(np.corrcoef(z_sim_sample[:,i],z_emb[:,i])[10:,:10])
fig.show()


time_files = os.listdir("../analysis_results/compiled/by_time")

if ".DS_Store" in time_files:
    time_files.remove(".DS_Store")

t_range = np.arange(1,299,2)

sim_ids_t = []
ave_d_range = np.zeros((len(sim_ids), len(t_range)))
for i, t in enumerate(t_range):
    try:
        dfi = pd.read_csv("../analysis_results/compiled/by_time/%.2f.csv" % t)
        ave_d_range[:, i] = dfi["av_av_AVE_d"]

    except:
        ave_d_range[:, i] = np.nan
    if i == 0:
        sim_ids_t = dfi["sim_index"]

for i in range(int(1000)):
    plt.scatter(t_range,ave_d_range[int(np.random.random()*len(sim_ids))].T)
plt.show()
