import numpy as np

def init_NYUv2():
    global f
    global D
    global Df
    global ps
    global MAX_KERNEL_SIZE
    
    f = 50e-3 # 50 mm --> m
    D = f / 8
    Df = np.array([1, 1.5, 2.5, 4, 6], dtype=np.float32)  # m
    ps = 1.2e-5 # m
    MAX_KERNEL_SIZE = 7