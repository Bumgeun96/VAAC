import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
dir = "./pickle_file"
algorithm = []
for n in range(len(os.listdir(dir))):
    algorithm.append(os.listdir(dir)[n].split(',')[1])
x = np.linspace(0, 500000, 50000)
y_limit = 0
for algo in algorithm:
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
plt.xlabel('steps')
plt.ylabel('Mean Number of Visited Blocks')
plt.legend()
plt.grid(True)
plt.show()