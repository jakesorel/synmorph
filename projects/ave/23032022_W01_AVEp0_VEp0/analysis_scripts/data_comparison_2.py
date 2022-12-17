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

df_embryo_full = pd.read_csv("../reference/single_cell_statistics_table_export-ecc-realxyz-from-demons_vcorr_455-Emb2-225-polar.csv")

quadrants = np.arange(1,17)
frames = np.unique(df_embryo_full["Frame"])
area_values = np.zeros((len(quadrants),len(frames)))
counts = np.zeros((len(quadrants),len(frames)))
for i, q in enumerate(quadrants):
    for j, f in enumerate(frames):
        dfi = df_embryo_full[(df_embryo_full["Static_Epi_Contour_Quad_ID"]==q)*(df_embryo_full["Frame"]==f)]
        area_values[i,j] = dfi["Area"].mean()
        counts[i,j] = dfi["Area"].count()

plt.plot(area_values[0]/area_values[:8].mean(axis=0))
plt.show()

###In the real dataset (1,0) is the direction of the AVE.

df_embryo = pd.read_csv("../reference/L455_E3.csv")
df_embryo = df_embryo[(df_embryo["Static_Quad_ID"]>0)*(df_embryo["Static_Quad_ID"]<17)]
df_embryo["ecc"] = np.sqrt(np.square(df_embryo['Major_Length']/2) - np.square(df_embryo['Minor_Length']/2)) / (df_embryo['Major_Length']/2)
df_embryo["N_neighbours"] = [len(trk.split(":")) for trk in df_embryo["Cell_Neighbours_Track_ID"]]
df_embryo = df_embryo[df_embryo["Stage"] == 'Migration to Boundary']

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

file_names = os.listdir("../analysis_results/compiled/by_position_class")
if ".DS_Store" in file_names:
    file_names.remove(".DS_Store")
sim_ids = [int(fl.split("_VEp0_")[1].split("_analysis")[0]) for fl in file_names]
sim_csv_files = ["../analysis_results/compiled/by_position_class/" + fl for fl in file_names]

moving_sim_ids = np.array([ 9422,  8781,  9489,  8997,  9205, 10213,  9424,  9410,  9426,
        8778,  7775,  9207,  9860,  8917,  9209,  9061,  9354,  9344,
        8486, 10359, 10294,  8630,  9419,  9409,  9071,  9503, 10278,
        9428,  8635, 10280,  9210,  9998,  9934,  9932,  9143,  9710,
        9064,  9856,  8423,  8273,  8346,  9645,  9208,  9212,  9416,
        9279,  8850, 10201,  9284,  8202, 10273,  8413,  8627,  8412,
        9214,  8636, 10000,  8922,  9719,  8563,  8135, 10291,  9859,
       10365,  8558, 10001,  9337,  9931, 10363, 10003, 10077,  8779,
       10078,  8567,  8564, 10071,  9427,  8130,  9213, 10142,  9415,
        9718,  8349,  9918,  9423, 10295, 10351,  9283,  8275,  9420,
       10074,  8058,  8418,  9991, 10362, 10206,  9780,  9854, 10070,
        8851,  8998,  8203,  8634,  4246,  8348, 10069, 10076,  9358,
       10145,  9137,  9500, 10150,  9277, 10286,  9996,  9499,  8417,
       10007, 10149,  8709,  8134,  8927,  8345,  8628,  8625,  9484,
       10140,  8421,  8926,  8923,  8347,  9357,  8633,  8488,  9791,
       10366, 10004,  8279,  8059,  9490,  9495,  9202, 10219, 10006,
        9636, 10354,  9493,  9271,  9343, 10002,  8422,  9140,  9356,
        7052,  9274,  8632,  9641, 10143,  8490, 10356,  9703,  8637,
       10353,  9067,  2583,  8701,  9142,  9348,  8639,  9642,  8631,
        9501,  9863,  8626,  9935,  9347,  9491, 10279, 10062, 10361,
       10364,  8415,  9990,  9070,  9714,  9852,  9483,  3378,  8924,
        9783,  9488, 10129,  9431, 10360,  8269,  8920,  8272,  8205,
       10287,  8341,  9351,  9646, 10148,  9413,  9429,  9482, 10217,
       10079,  9139, 10222,  9211,  9199,  8487,  9643,  9708, 10073,
        8494,  8988,  9571,  9784,   788,  9285,  2591,  8493,  9215,
        9069, 10358,  9430,  9345,  9789,  9933,  8562,  9342,  9487,
        8278,  9498,  9573,  9481, 10075, 10072,  9492,  9926,  9134,
        9924,  9790, 10352,  9286,  9204,  9352,  9135,  8566,  8062,
        8565,  9282,  8277,  8342,  9717,  8131,  9417, 10350, 10141,
        9276,  8206, 10137, 10220, 10281,  9930,  9997,  8853,  8344,
       10282,  8551, 10064,  9574,  9502,  8993,  8629, 10147, 10216,
        8557,  9421, 10367, 10146,  9496,  8061, 10223,  8491,  8063,
       10289,  9919,  9066,  8350,  7767, 10063, 10293,  9475,  7990,
       10005,  9346,  8419, 10284, 10211, 10215, 10344, 10288, 10212,
        9709, 10221,  9497,  9349,  8495,  8489, 10135,  9355,  9425,
        8999,  9132,  7483,  8561,  8989,  8414,  9206,  9272,  9065,
        9287,  8707,  8492,  9136,  9353,  9486,  9494,  4317, 10283,
        9127, 10290,  8351, 10134,  9715,  8420, 10218,  9350,  9278,
        7411,  9200,  9138,  9927,  9781, 10136,  9570,  9785,  9858,
        9862,  9925,  8994, 10151,  8276,  8274,  9788,  8995, 10214,
        8845,  8783,  6980, 10285,  8925,  9786,  8706, 10292, 10345,
        9418,  9572,  8638, 10068, 10357,  9861,  9787,  9853, 10065,
        9141,  8485,  9359])

df_sim = pd.read_csv(sim_csv_files[0])
column_ids = [list(df_sim.columns).index(c) for c in columns_simulation]
frames_sim = np.unique(df_sim["t"])
q=0
simulation_statistics_by_quadrant_by_simulation = np.zeros((len(sim_ids),len(frames_sim),16,len(column_ids)))
for i, (sim_id, sim_csv_file) in enumerate(zip(sim_ids,sim_csv_files)):
    print(q)
    if sim_id in moving_sim_ids:
        df_sim = pd.read_csv(sim_csv_file)
        simulation_statistics_by_quadrant = df_sim.values.T[column_ids].T.reshape(len(frames_sim),16,len(column_ids))
        simulation_statistics_by_quadrant_by_simulation[i] = simulation_statistics_by_quadrant
    else:
        simulation_statistics_by_quadrant_by_simulation[i] = np.nan
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


def corr(sim,emb):
    sim_mean = sim.mean(axis=1)
    emb_mean = emb.mean(axis=1)
    dsim = sim - np.expand_dims(sim_mean,1)
    demb = emb - np.expand_dims(emb_mean,1)
    dsim = np.expand_dims(dsim,1)
    demb = np.expand_dims(demb,0)
    numerator = (dsim*demb).sum(axis=3)
    denominator = np.sqrt((dsim**2).sum(axis=3)*(demb**2).sum(axis=3))
    return numerator/denominator


def corr(_sim,_emb):
    sim = np.expand_dims(_sim,1)
    emb = np.expand_dims(_emb,0)
    n = sim.shape[-1]
    numerator = n*np.sum(sim*emb,axis=-1) - np.sum(sim,axis=-1)*np.sum(emb,axis=-1)
    denominator = np.sqrt((n*np.sum(sim**2,axis=-1) - np.sum(sim,axis=-1)**2)*(n*np.sum(emb**2,axis=-1) - np.sum(emb,axis=-1)**2))
    return numerator/denominator



nt_sim = simulation_statistics_by_quadrant_by_simulation.shape[1]
nt_emb = embryo_statistics_by_quadrant.shape[0]


log2_grad_lim = -1,1
n_grads = 10

masks = np.zeros((n_grads,nt_sim,nt_emb))

Tsim,Temb = np.meshgrid(np.arange(nt_sim),np.arange(nt_emb),indexing="ij")
for i, log2_grad in enumerate(np.linspace(log2_grad_lim[0],log2_grad_lim[1],n_grads)):
    grad = 2**(log2_grad)
    mask = 1/(1 + (Temb - Tsim*grad)**2)
    mask /= mask.sum()
    mask /= mask.size
    masks[i] = mask




j = 1223

emb = embryo_statistics_by_quadrant.copy()

corrs = np.zeros(len(simulation_statistics_by_quadrant_by_simulation))
for i, sim in enumerate(simulation_statistics_by_quadrant_by_simulation):
    corrs[i] = (np.expand_dims(masks,3)*np.expand_dims(corr(sim,emb),0)).sum(axis=(1,2,3)).max()
    print(i)




W01_range = np.unique(W01)
AVEp0_range = np.unique(AVE_p0)
VEp0_range = np.unique(nonAVE_p0)

cost_sim_square = corrs[np.argsort(sim_ids)].reshape(12,12,12,6)
cost_sim_square_average = np.nanmean(cost_sim_square,axis=-1)

W01_square = W01[np.argsort(sim_ids)].reshape(12,12,12,6).mean(axis=-1)
VEp0_square = nonAVE_p0[np.argsort(sim_ids)].reshape(12,12,12,6).mean(axis=-1)

# c = stats.rankdata(cost_sim_square_average.ravel(),"average").reshape(cost_sim_square_average.shape)/cost_sim_square_average.size

# c = (np.argsort(cost_sim_square_average.ravel())/len(cost_sim_square_average.ravel())).reshape(cost_sim_square_average.shape)

for i in range(12):
    fig, ax = plt.subplots()
    ax.imshow(cost_sim_square_average[:,:,i],vmin=np.nanmin(cost_sim_square_average),vmax=np.nanmax(cost_sim_square_average),interpolation=None)
    fig.savefig("vep0%d.pdf"%i,dpi=300)
plt.close("all")


for i in range(12):
    fig, ax = plt.subplots()
    ax.imshow(cost_sim_square_average[:,i],vmin=np.nanmin(cost_sim_square_average),vmax=np.nanmax(cost_sim_square_average),interpolation=None)
    fig.savefig("avep0%d.pdf"%i,dpi=300)
plt.close("all")




titles = ["Sim\ninner8","Data\ninner8","Sim\nouter8","Data\nouter8"]
interp = "bicubic"
cmap = plt.cm.inferno
fig,ax = plt.subplots(7,4,figsize=(8,14),sharex=True,sharey=True)
j = np.argsort(corrs)[-1000]
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


fig, ax = plt.subplots(7,1)
ax = ax.ravel()
for i in range(7):
    ax[i].imshow(np.corrcoef(simulation_statistics_by_quadrant_by_simulation[j,:,i],embryo_statistics_by_quadrant[:,i])[100:,:100])
fig.savefig("correlations.pdf")

