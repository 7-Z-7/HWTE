import pickle
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

path="../../data/embedding_table.pickle"
with open(path,'rb') as f:
    data=pickle.load(f)
print(len(data))
table=data[112:]
randomized_pca = PCA(n_components=2, svd_solver='randomized')
reduced_data_rpca = randomized_pca.fit_transform(table)
pca = PCA(n_components=2)
reduced_data_pca = pca.fit_transform(table)

# colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
jet = cm = plt.get_cmap('jet')

NCURVES=3000
values = range(NCURVES)
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

plt.figure()
for i,one in enumerate(reduced_data_rpca):
    colorVal = scalarMap.to_rgba(values[i])
    plt.scatter(one[0], one[1],color=colorVal)
# plt.scatter(reduced_data_rpca[:,0], reduced_data_rpca[:,1])
plt.show()

# new_data=[]
# new_30_data=[]
# for one in data:
#     if one>0:
#         new_data.append(one)
#         if one<=30:
#             new_30_data.append(one)
# data=new_data
# # plt.figure()
# # # plt.hist(data,bins=100,range=[1,100],log=True)
# # # plt.hist(data,bins=100,cumulative=True,)
# # plt.hist(data,bins=100)
# # plt.show()
# # print(np.mean(data))
#
# print(np.mean(data))
# print(len(data))
# print(len(new_30_data))
# print(len(new_30_data)/len(data))
# exit()
#
# fig, ax1 = plt.subplots()
# ax1.grid(True)
# ax1.hist(data,bins=100,label='number')
# ax1.set_ylabel('number')
#
# ax2 = ax1.twinx()
# ax2.hist(data,bins=100, density=True, histtype='step',cumulative=True,label='percent',color='r')
# ax2.set_ylabel('percent')
#
#
# fig.legend(loc='best',bbox_to_anchor=(0.4, 0.1, 0.5, 0.5))
# # ax2.legend(loc='right')
# ax1.set_xlabel('length')
# ax1.set_title("Length distribution of session routes in one day")
#
#
# # fig.tight_layout()
# plt.show()

