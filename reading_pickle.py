import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("./pickle_file/map:1,sac,visitation_plot.pickle", "rb") as f:
    base_line_data = pickle.load(f)
    
with open("./pickle_file/map:1,iaac,visitation_plot.pickle", "rb") as f:
    data = pickle.load(f)
    
base_line_data = np.array(base_line_data)
data = np.array(data)
x = np.linspace(0, 300000, 30000)
plt.plot(x, base_line_data.mean(axis = 0), label='SAC')
plt.fill_between(x, base_line_data.mean(axis = 0)-base_line_data.std(axis = 0), base_line_data.mean(axis = 0)+base_line_data.std(axis = 0), alpha=0.3)
plt.plot(x, data.mean(axis = 0), label='IAAC')
plt.fill_between(x, data.mean(axis = 0)-data.std(axis = 0), data.mean(axis = 0)+data.std(axis = 0), alpha=0.3)
plt.xlabel('steps')
plt.ylabel('Mean Number of Visited Blocks')
plt.legend()
plt.grid(True)
plt.show()