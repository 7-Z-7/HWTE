import pickle
import matplotlib.pyplot as plt
import numpy as np

path='inform_of_tf_during_train_data_no0.pickle'

with open(path,'rb') as f:
    data=pickle.load(f)

print(type(data[0]))
print(type(data[1]))

times=sorted(list(data[0].values()),reverse=True)[1:]

days=data[1]
print(np.mean(days))
print(np.min(days))
print(np.max(days))
plt.hist(days,100)
# plt.violinplot(days)
# plt.hist(times,1000)
# plt.bar(range(len(times)),times)
plt.show()