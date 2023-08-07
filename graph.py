import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import time

def data_process(data, size):
	means = []
	x = []
	stds = []
	for i in range(len(data)):
		means.append(np.mean(data[max(i - size, 0):i + 1]))
		x.append(i)
		stds.append(np.std(data[max(i - size, 0):i + 1]))
	stds = np.array(stds)

	return x, means, stds

def data_fill(data):
	temp = np.array(data[-20:])
	while len(data) < 1000:
		data += temp[np.random.randint(20, size=20)].tolist()
	return data[:1000]

env = 'Hopper-v3'
env = 'Hopper-v2'
env = 'SparseHopper-v1'
# env = 'SparseHalfCheetah-v1'
# env = 'HalfCheetah-v3'
# env = 'Walker2d-v3'
seed = 0

def data_smoothing(data,window_portion = 0.1):
    temp = []
    window_portion = max(min(1,window_portion),0)
    window_size = int(data.shape[1]*window_portion)
    for i in range(data.shape[1]):
        temp.append(data[0][max(i+1-window_size,0):i+1].mean())
    return np.array([temp])

plt.title(env,fontsize=20)
if os.path.isfile("./figures/returns_"+env+".pickle"):
	with open("./figures/returns_"+env+".pickle","rb") as fr:
		dictionary = pickle.load(fr)
		return_values = []
		for i,seed in enumerate(list(dictionary.keys())):
			if not (i==0) and not(len(return_values[0])-len(dictionary[seed]) == 0):
				return_values.append(np.concatenate((np.array(dictionary[seed]),np.zeros(len(return_values[0])-len(dictionary[seed]))),axis=0))
			else:
				return_values.append(np.array(dictionary[seed]))
		return_values = np.array(return_values)
		return_values = data_smoothing(return_values,0.1)
		mean = return_values.mean(axis=0)
		std = return_values.std(axis=0)
		upper_bound = mean+std
		lower_bound = mean-std

Range = np.arange(0.001,10,0.001)
Range = Range[:len(mean)]
plt.plot(Range,mean)
plt.fill_between(Range,lower_bound, upper_bound, alpha=0.25)

ax = plt.gca()
ax.set_facecolor('#E7EAEF')
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tick_params(top=False, bottom=False, left=False, right=False)
plt.grid(True,color='white',alpha=1)
plt.xlabel('steps',fontsize=15)
plt.ylabel('average return',fontsize=15)
plt.legend(loc='upper left',fontsize=13)
plt.subplots_adjust(left=0.12, right=0.97, top=0.935, bottom=0.110)

# plt.legend(loc='lower right')
plt.savefig(env+'.png',dpi=600)
plt.close()
# plt.show()
