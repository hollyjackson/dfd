import numpy as np

global AVG_SENSOR_WIDTH
AVG_SENSOR_WIDTH = 3.1 * 1e-3 # m
global window_size

def init_NYUv2():
    global f
    global D
    global Df
    global ps
    global MAX_KERNEL_SIZE
    global thresh
    global min_Z
    global max_Z
    
    f = 50e-3 # 50 mm --> m
    D = f / 8
    Df = np.array([1, 1.5, 2.5, 4, 6], dtype=np.float32)  # m
    ps = 1.2e-5 # m (3.1e-3 / 240)
    # sensor_width (m) / image width (pixels)
    MAX_KERNEL_SIZE = 7

    min_Z = 0.1
    max_Z = 10. # m
    thresh = 2

def init_DefocusNet():
    global f
    global D
    global Df
    global ps
    global MAX_KERNEL_SIZE
    global thresh
    global min_Z
    global max_Z

    Df = np.array([0.1, 0.15, 0.3, 0.7, 1.5], dtype=np.float32) # m
    f = 2.9 * 1e-3 # 2.9 mm --> m  
    D = f / 1. # f-number = 1.
    ps = 1.2e-5 #36e-3 / 256.
    MAX_KERNEL_SIZE = 7

    min_Z = 0.1
    max_Z = 3 # m
    thresh = 0.5

def init_MobileDepth():
    global f
    global D
    global Df
    global ps
    global MAX_KERNEL_SIZE
    global thresh
    global min_Z
    global max_Z

    MAX_KERNEL_SIZE = 7
    ps = 1 # all things are unitless / already in "pixels" from calib data
#     
    # these are approximate settings using by Si  
    # f = 50 * 1e-3
    # D = f / 24. # fnumber = 24
    # ps = 5.6e-6
    # min_Z = 0.1
    # max_Z = 10. # maybe use 50 inches idk?

def init_Make3D():
    global f
    global D
    global Df
    global ps
    global MAX_KERNEL_SIZE
    global thresh
    global min_Z
    global max_Z

    min_Z = 0.01 # ~0.1
    max_Z = 80 # or 70 m
    
    Df = np.array([1, 2, 4, 8, 16, 32, 64], dtype=np.float32)
    print('Df:',Df)

def init_KITTI():

    ps = 4.65e-6 # https://avsupply.com/ITM/29755/fl2-14s3c-c.html/
