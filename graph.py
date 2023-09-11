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

def data_smoothing(data,window_portion = 0.1):
    window_portion = max(min(1,window_portion),0)
    window_size = int(data.shape[1]*window_portion)
    num_rows, num_cols = data.shape
    smoothed_data = np.zeros_like(data, dtype=float)
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            start_col = max(col_idx + 1 - window_size, 0)
            end_col = col_idx + 1
            smoothed_data[row_idx, col_idx] = np.mean(data[row_idx, start_col:end_col])
    return smoothed_data

env = ['SparseHopper-v1','SparseAnt-v1','SparseHalfCheetah-v1','SparseWalker2d-v1']
seed = 0
fig, ax = plt.subplots(2, 2, figsize=(20, 8))
ax = ax.ravel()

for i,e in enumerate(env):
    try:
        if os.path.isfile("./figures/returns_"+e+".pickle"):
            with open("./figures/returns_"+e+".pickle","rb") as fr:
                dictionary = pickle.load(fr)
                return_values = []
                for k,seed in enumerate(list(dictionary.keys())):
                    if not (k==0) and not(len(return_values[0])-len(dictionary[seed]) == 0):
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
                ax[i].plot(Range,mean)
                ax[i].fill_between(Range,lower_bound, upper_bound, alpha=0.25)
                ax[i].set_title(e,fontsize=20)
                ax[i].set_facecolor('#E7EAEF')
                ax[i].tick_params(top=False, bottom=False, left=False, right=False)
                ax[i].grid(True, color='white', alpha=1)
                ax[i].set_xlabel('Steps', fontsize=15)
                ax[i].set_ylabel('Average Return', fontsize=15)
    except:
        pass
plt.legend(loc='upper left',fontsize=13)
plt.subplots_adjust(left=0.12, right=0.97, top=0.935, bottom=0.110)
plt.savefig('Sparse.png',dpi=600)
plt.close()

x = np.linspace(0, 1000000, 1000)
algorithm = ['vaac','sac','rnd']
for e in env:
    y_limit = 0
    for algo in algorithm:
        try:
            with open('./pickle_file/map:'+e+","+algo+".pickle", "rb") as f:
                data = np.array(pickle.load(f))
                data = data_smoothing(data,0.1)
                plt.plot(x,data.mean(axis=0)[-1000:],label=algo.upper())
                plt.fill_between(x, data.mean(axis = 0)[-1000:]-data.std(axis = 0)[-1000:], data.mean(axis = 0)[-1000:]+data.std(axis = 0)[-1000:], alpha=0.3)
                y_limit = max(y_limit,max(data.mean(axis = 0)+data.std(axis = 0)))
                mean_max = np.max(data.mean(axis=0))
                arg = np.argmax(data.mean(axis=0))
                print('Env:'+e+', Algorithm:'+algo+', Mean:'+str(round(mean_max,1))+', Std:'+str(round(data.std(axis = 0)[arg],1)))
        except:
            pass
    plt.xlim(x[0],x[-1])
    plt.ylim(0,y_limit*1.1)
    plt.legend(loc='upper left',fontsize = 15)
    plt.grid(True)
    plt.subplots_adjust(left=0.12, right=0.95, top=0.97, bottom=0.1)
    plt.savefig(e+'.png',dpi=600)
    plt.close()