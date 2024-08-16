import os
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates
import torch

def gpu_auto(type = 'gpu'):
    memory = []
    gpu = []
    nvmlInit()
    num_gpu = torch.cuda.device_count()
    for i in range(num_gpu):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        utilization = nvmlDeviceGetUtilizationRates(handle)
        memory_percent = info.total / (1024**2)
        gpu_percent = utilization.gpu
        memory.append(memory_percent)
        gpu.append(gpu_percent)
    nvmlShutdown()
    if type == 'gpu':
        temp = gpu
    elif type == 'memory':
        temp = memory
    idx = temp.index(min(temp))
    print('GPU: ',idx)
    return idx