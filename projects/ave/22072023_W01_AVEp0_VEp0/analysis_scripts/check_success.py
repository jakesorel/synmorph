import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd



files = open("../scan_summary/all_files.txt").readlines()
numbers = [int(nm.split("_")[-2]) for nm in files]

N = 10
W01, AVE_p0, VE_p0, AVE_v0, lambda_P, seed = np.mgrid[:N,:N,:N,:N,:N,:N]
index = np.zeros_like(W01)
for i, val in enumerate(np.flip([W01, AVE_p0, VE_p0, AVE_v0, lambda_P, seed])):
    index += 10**i * val

indexed = np.zeros(int(1e6),dtype=bool)
for number in numbers:
    indexed[number] = True

indexed = indexed.reshape(W01.shape)

not_indexed = list(set(list(index.ravel())).difference(set(numbers)))
fl = open("../scan_summary/not_run.txt","w")
fl.write("\n".join(np.array(not_indexed).astype(str)))
fl.close()
#
# df = pd.DataFrame({"W01":W01.ravel()[numbers],
#                    "AVE_p0":AVE_p0.ravel()[numbers],
#                    "VE_p0":VE_p0.ravel()[numbers],
#                    "AVE_v0":AVE_v0.ravel()[numbers],
#                    "lambda_P":lambda_P.ravel()[numbers],
#                    "seed":seed.ravel()[numbers]})
#
# sns.pairplot(data=df)
# plt.show()
#
#
# fig, ax = plt.subplots(5,5)
# for i in range(5):
#     for j in range(5):
#         if i > j:
#             ax[j,i].imshow(indexed.sum(axis=tuple(set(np.arange(6)).difference([i,j]))),vmin=0,vmax=10000)
# fig.show()
#
#
# """
#     W01 = W01_range[i1]
#     AVE_p0 = AVE_p0_range[i2]
#     VE_p0 = VE_p0_range[i3]
#     AVE_v0 = AVE_v0_range[i4]
#     lambda_P = lambda_P_range[i5]
#     seed = seed_range[j]
# """