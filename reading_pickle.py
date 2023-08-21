import pickle
import numpy as np
import matplotlib.pyplot as plt
algorithm = ['random','sac','vaac']
x = np.linspace(0, 300000, 30000)
for algo in algorithm:
    with open("./pickle_file/map:1,"+algo+",visitation_plot.pickle", "rb") as f:
        data = np.array(pickle.load(f))
        plt.plot(x, data.mean(axis = 0), label=algo.upper())
        plt.fill_between(x, data.mean(axis = 0)-data.std(axis = 0), data.mean(axis = 0)+data.std(axis = 0), alpha=0.3)

plt.xlim(x[0],x[-1])
plt.xlabel('steps')
plt.ylabel('Mean Number of Visited Blocks')
plt.legend()
plt.grid(True)
plt.show()