import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_type',type=str,default='max_mean')
    args = parser.parse_args()
    return args

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

def data_smoothing(data,window_portion = 0.05,window_size = None):
    if window_size == None:
        window_portion = max(min(1,window_portion),0)
        window_size = int(data.shape[1]*window_portion)
    else:
        pass
    num_rows, num_cols = data.shape
    smoothed_data = np.zeros_like(data, dtype=float)
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            start_col = max(col_idx + 1 - window_size, 0)
            end_col = col_idx + 1
            smoothed_data[row_idx, col_idx] = np.mean(data[row_idx, start_col:end_col])
    return smoothed_data

args = parse_args()
env = ['SparseHopper-v1','SparseAnt-v1','SparseHalfCheetah-v1','SparseWalker2d-v1','HumanoidStandup-v1','Humanoid-v1','DelayedHopper-v1','DelayedAnt-v1','DelayedHalfCheetah-v1','DelayedWalker2d-v1',]
seed = 0
fig, ax = plt.subplots(5, 2, figsize=(20, 12))
ax = ax.ravel()
print('============================================')
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
                return_values = data_smoothing(return_values,0.05,window_size=20)
                mean = return_values.mean(axis=0)
                std = return_values.std(axis=0)
                upper_bound = mean+std
                lower_bound = mean-std
                Range = np.arange(0.001,20,0.001)
                Range = Range[:len(mean)]
                ax[i].plot(Range,mean)
                ax[i].fill_between(Range,lower_bound, upper_bound, alpha=0.25)
                ax[i].set_title(e,fontsize=20)
                ax[i].set_facecolor('#E7EAEF')
                ax[i].tick_params(top=False, bottom=False, left=False, right=False)
                ax[i].grid(True, color='white', alpha=1)
                ax[i].set_xlabel('Steps', fontsize=15)
                ax[i].set_ylabel('Average Return', fontsize=15)
                
                if args.plot_type == 'mean_max': 
                    print(round(np.max(return_values[:-1].mean(axis=0)),1),round(return_values[:-1].std(axis = 0)[np.argmax(return_values[:-1].mean(axis=0))],1))
                    mean_max = np.max(return_values.mean(axis=0))
                    arg = np.argmax(return_values.mean(axis=0))
                    print('Env:'+e+', Mean:'+str(round(mean_max,1))+', Std:'+str(round(return_values.std(axis = 0)[arg],1))+' # seed: '+str(len(return_values)))
                elif args.plot_type == 'max_mean':
                    print(round(return_values[:-1].max(axis=1).mean(),1),round(return_values[:-1].max(axis = 1).std(),1))
                    print('Env:'+e+', Mean:'+str(round(return_values.max(axis=1).mean(),1))+', Std:'+str(round(return_values.max(axis = 1).std(),1))+' # seed: '+str(len(return_values)))
    except:
        pass
print('============================================')
plt.legend(loc='upper left',fontsize=13)
plt.subplots_adjust(left=0.12, right=0.97, top=0.935, bottom=0.110)
plt.savefig('Sparse.png',dpi=600)
plt.close()

algorithm = ['vaac','sac','rnd']
for e in env:
    y_limit = 0
    for algo in algorithm:
        try:
            with open('./pickle_file/map:'+e+","+algo+".pickle", "rb") as f:
                data = np.array(pickle.load(f))
                data = data_smoothing(data,0.05,window_size=20)
                x = np.linspace(0,1000*data.shape[1],1000)
                try:
                    plt.plot(x,data.mean(axis=0)[-data.shape[1]:],label=algo.upper())
                    plt.fill_between(x, data.mean(axis = 0)[-data.shape[1]:]-data.std(axis = 0)[-data.shape[1]:], data.mean(axis = 0)[-data.shape[1]:]+data.std(axis = 0)[-data.shape[1]:], alpha=0.3)
                except:
                    pass
                y_limit = max(y_limit,max(data.mean(axis = 0)+data.std(axis = 0)))
                if args.plot_type == 'mean_max': 
                    mean_max = np.max(data.mean(axis=0)[50:])
                    arg = np.argmax(data.mean(axis=0)[50:])
                    print('Env:'+e+', Algorithm:'+algo+', Mean:'+str(round(mean_max,1))+', Std:'+str(round(data.std(axis = 0)[arg],1)))
                elif args.plot_type == 'max_mean':
                    max_data = data[:,50:].max(axis=1)
                    print('Env:'+e+', Algorithm:'+algo+', Mean:'+str(round(max_data.mean(),1))+', Std:'+str(round(max_data.std(),1)))
        except:
            pass
    plt.xlim(x[0],x[-1])
    plt.ylim(0,y_limit*1.1)
    plt.legend(loc='upper left',fontsize = 15)
    plt.grid(True)
    plt.subplots_adjust(left=0.12, right=0.95, top=0.97, bottom=0.1)
    plt.savefig(e+'.png',dpi=600)
    plt.close()

for e in env:
    try:
        fig, ax = plt.subplots(5,1, figsize=(20, 12))
        ax = ax.ravel()
        with open("./figures/returns_"+e+".pickle","rb") as fr:
            dictionary = pickle.load(fr)
            return_values = []
            xlim = 0
            for k,seed in enumerate(list(dictionary.keys())):
                data = np.array(dictionary[seed])
                length = len(data)
                Range = np.arange(0.001,20,0.001)
                Range = Range[:length]
                if xlim<Range[-1]:
                    xlim = Range[-1]
                ax[k].plot(Range,data)
                ax[k].set_xlim([0,xlim])
        plt.subplots_adjust(left=0.12, right=0.97, top=0.935, bottom=0.110)
        plt.savefig('./seedplot/'+e+'.png',dpi=600)
        plt.close()
    except:
        pass
    
# max_list = []
# with open("./figures/returns_SparseAnt-v1 (copy).pickle","rb") as fr:
#     dictionary = pickle.load(fr)
#     for k,seed in enumerate(list(dictionary.keys())):
#         max_list.append(data_smoothing(np.array([dictionary[seed]]),0.05).max())
# with open("./figures/returns_SparseAnt-v1 (another copy).pickle","rb") as fr:
#     dictionary = pickle.load(fr)
#     for k,seed in enumerate(list(dictionary.keys())):
#         max_list.append(data_smoothing(np.array([dictionary[seed]]),0.05).max())
# print(np.array(max_list).mean())
# print(np.array(max_list).std())


with open("./pickle_file/map:SparseHalfCheetah-v1,vaac (seed:10~50).pickle","rb") as fr:
    data1 = np.array(pickle.load(fr))
with open("./pickle_file/map:SparseHalfCheetah-v1,vaac.pickle","rb") as fr:
    data2 = np.array(pickle.load(fr))
data = np.vstack((data1,data2))
data = data_smoothing(data,0.05,window_size=10)
max_data = data[:,50:].max(axis=1)
print(max_data.mean(),max_data.std())