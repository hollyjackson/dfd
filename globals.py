import torch

def init_NYUv2():
    global f
    global D
    global Df
    global ps
    global MAX_KERNEL_SIZE
    
    f = 50e-3 # 50 mm --> m
    D = f / 8
    Df = torch.tensor([1, 1.5, 2.5, 4, 6])  # m
    ps = 1.2e-5 # m
    MAX_KERNEL_SIZE = 7