import numpy as np

AVG_SENSOR_WIDTH = 3.1 * 1e-3 # m

def init_NYUv2():
    global f
    global D
    global Df
    global ps
    global MAX_KERNEL_SIZE
    global thresh
    
    f = 50e-3 # 50 mm --> m
    D = f / 8
    Df = np.array([1, 1.5, 2.5, 4, 6], dtype=np.float32)  # m
    ps = 1.2e-5 # m (3.1e-3 / 240)
    # sensor_width (m) / image width (pixels)
    MAX_KERNEL_SIZE = 7

    # min_Z = 0.1
    # max_Z = 10. # m
    # thresh = 2

def init_DefocusNet():
    global f
    global D
    global Df
    global ps
    global MAX_KERNEL_SIZE
    global thresh

    Df = np.array([0.1, 0.15, 0.3, 0.7, 1.5], dtype=np.float32) # m
    f = 2.9 * 1e-3 # 2.9 mm --> m  
    D = f / 1. # f-number = 1.
    ps = 1.2e-5 # 3.1e-3 / 256
    MAX_KERNEL_SIZE = 7

    # min_Z = 0.1
    # max_Z = 3 # m
    thresh = 0.5

# def init_MobileDepth():
#     global f
#     global D
#     global Df
#     global ps
#     global MAX_KERNEL_SIZE
#     
#     # these are approximate settings using by Si  
#     f = 50 * 1e-3
#     D = f / 24. # fnumber = 24
#     ps = 5.6e-6
#     min_Z = 0.1
#     max_Z = 10. # maybe use 50 inches idk?