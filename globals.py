import numpy as np

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
    ps = 1.2e-5 # m (3.1e-3 / 240) -- sensor_width (m) / image width (pixels)

    min_Z = 0.1
    max_Z = 10. # m
    thresh = 2

def init_MobileDepth():
    global f
    global D
    global Df
    global ps
    global MAX_KERNEL_SIZE
    global thresh
    global min_Z
    global max_Z

    thresh = 0.1
    ps = 0.75 * 2

    min_Z = 1
    max_Z = 800

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
    max_Z = 80 #
    
    Df = np.array([1, 2, 4, 8, 16, 32, 64], dtype=np.float32)

    thresh = 0.5

