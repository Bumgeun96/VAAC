import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from plotlib import plot_visiting
from ContinuousGridworld.wall_setting import map_1

dir = "./pickle_file"
algorithm = []
for n in range(len(os.listdir(dir))):
    algorithm.append(os.listdir(dir)[n].split(',')[1])
algorithm = sorted(list(set(algorithm)),key=lambda x: x[0], reverse= True)
algorithm = ['vaac','sac','rnd']
print(algorithm)
x = np.linspace(0, 500000, 50000)
y_limit = 0
for algo in algorithm:
    if algo == 'random':
        continue
    with open(dir+"/map:1,"+algo+",visitation_plot.pickle", "rb") as f:
        data = np.array(pickle.load(f))
        try:
            plt.plot(x, data.mean(axis = 0), label=algo.upper())
            plt.fill_between(x, data.mean(axis = 0)-data.std(axis = 0), data.mean(axis = 0)+data.std(axis = 0), alpha=0.3)
            y_limit = max(y_limit,max(data.mean(axis = 0)+data.std(axis = 0)))
        except:
            pass

plt.xlim(x[0],x[-1])
plt.ylim(0,y_limit*1.1)
plt.xlabel('Time Steps',fontsize = 15)
plt.ylabel('Mean Number of Visited Blocks',fontsize = 15)
plt.legend(loc='upper left',fontsize = 15)
plt.grid(True)
plt.subplots_adjust(left=0.12, right=0.95, top=0.97, bottom=0.1)
plt.savefig('visitation.png',dpi=600)
plt.close()
env_wall, _ = map_1()
fig, ax = plt.subplots(3, 3, figsize=(20, 8))
for n,algo in enumerate(algorithm):
    if algo == 'random':
        continue
    try:
        with open(dir+"/map:1,"+algo+",visiting_histogram.pickle", "rb") as f:
            data = np.array(pickle.load(f))
            per_name = ["5k", "50k", str(int(x[-1]/1000))+"k"]
            for i,t in enumerate(data):
                try:
                    ax[n][i].set_xlim((-0.5, 101 -0.5))
                    ax[n][i].set_ylim((101-0.5, -0.5))
                    im = ax[n][i].imshow(t, vmax = 10,cmap='hot')
                    walls = np.zeros([101, 101])
                    for w in env_wall:
                        if w != (0, 101 - 1):
                            walls[w] = 1
                        else:
                            walls[w] = None
                    ax[n][i].imshow(walls,cmap='Blues',alpha=0.25)
                    if i == 0:
                        ax[n][i].set_xticks([0,20,40,60,80,100])
                        ax[n][i].set_xticklabels([0,20,40,60,80,100])
                        ax[n][i].set_yticks([100,80,60,40,20,0])
                        ax[n][i].set_yticklabels([0,20,40,60,80,100])
                        ax[n][i].set_ylabel(algo.upper(),fontsize = 20)
                    else:
                        ax[n][i].set_xticklabels([])
                        ax[n][i].set_yticklabels([])
                    if n == 0:
                        ax[n][i].set_title(per_name[i], size=20)
                except:
                    pass
    except:
        pass
fig.subplots_adjust(left=0.05, right=1, top=0.97, bottom=0.05)
bar = fig.colorbar(im, ax=ax, shrink=1)
bar.set_ticks([0,2,4,6,8,10])
bar.set_ticklabels([0,2,4,6,8,'>'+str(10)])
fig.savefig("./result/visiting_histogram.png",dpi=600)