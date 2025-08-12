import os
import math

import numpy as np
import scipy
import matplotlib.pyplot as plt

from PIL import Image
import cv2

import torch
torch.cuda.empty_cache()
import torch_sparse

import utils
# from globals import *
import globals

def compute_u_v():
    max_kernel_size = globals.MAX_KERNEL_SIZE    
    # lim = int((float(max_kernel_size) - 1) / 2.)
    lim = max_kernel_size // 2

    us = torch.linspace(-lim, lim, max_kernel_size)
    vs = torch.linspace(-lim, lim, max_kernel_size)
    grid_u, grid_v = torch.meshgrid(us, vs, indexing='ij')
    u = grid_u.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    v = grid_v.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    return u, v

def compute_shifted_indices(width, height):
    max_kernel_size = globals.MAX_KERNEL_SIZE
    # lim = int((float(max_kernel_size) - 1) / 2.)
    lim = max_kernel_size // 2
    
    row_indices = torch.zeros((width,height,max_kernel_size,max_kernel_size))
    col_indices  = torch.zeros((width,height,max_kernel_size,max_kernel_size))
    
    grid = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='ij')
    indices = torch.stack(grid, dim=-1)

    for i in range(-lim, lim+1):
        for j in range(-lim, lim+1):
            row = indices[:,:,0] + i
            col = indices[:,:,1] + j
            row_indices[:,:,i+lim,j+lim] = row
            col_indices[:,:,i+lim,j+lim] = col

    return row_indices, col_indices

def generate_mask(row_indices, col_indices, width, height):
    
    condition1 = row_indices.flatten() < 0
    condition2 = row_indices.flatten() >= width
    condition3 = col_indices.flatten() < 0
    condition4 = col_indices.flatten() >= height
    indices_to_delete = torch.where(condition1 | condition2 | condition3 | condition4)
    mask = torch.ones(col_indices.flatten().size(0), dtype=torch.bool)
    mask[indices_to_delete] = False

    return mask

def compute_mask_flattened_indices(row_indices, col_indices, mask, width, height):
    max_kernel_size = globals.MAX_KERNEL_SIZE
    
    flattened_indices = row_indices * (height) + col_indices
    
    row = torch.tensor(range(width*height)).unsqueeze(1).repeat(1, max_kernel_size*max_kernel_size)
    row = row.flatten()
    row = row[mask].to(torch.int64)
    
    col = flattened_indices.flatten()
    col = col[mask].to(torch.int64)

    return row, col

def precompute_indices(width, height):#, max_kernel_size = 7):
    max_kernel_size = globals.MAX_KERNEL_SIZE
    
    u, v = compute_u_v()#max_kernel_size = max_kernel_size)
    row_indices, col_indices = compute_shifted_indices(width, height)#, max_kernel_size = max_kernel_size)
    # grab indices to remove 
    mask = generate_mask(row_indices, col_indices, width, height)
    row, col = compute_mask_flattened_indices(row_indices, col_indices, mask, width, height)#, max_kernel_size = max_kernel_size)

    return u, v, row, col, mask

# def forward_si(dpt, aif):
    
#     r = computer(dpt, globals.Df)
#     # r = r.unsqueeze(-1).unsqueeze(-1)
    
#     width, height, fs = r.shape

#     x, y = torch.meshgrid(
#         torch.arange(width),
#         torch.arange(height),
#         indexing='ij')

#     print(r.shape, y.shape, x.shape)

#     mid = globals.MAX_KERNEL_SIZE // 2

    
#     kernel_sum = torch.zeros((fs, width, height))
#     defocus_stack = torch.zeros((fs, width, height, 3))
#     # for n in range(fs):
#     #     defocus_stack[n] = aif.clone()
    
#     for n in range(fs):
#         sigma = r[:,:,n].squeeze()
#         mask1 = sigma > 1
#         # kernel_sum[n][sigma <= 1] = 1.
#         for i in range(globals.MAX_KERNEL_SIZE):
#             mask2 = ((x + i - mid) >= 0) & ((x + i - mid) < width)
#             for j in range(globals.MAX_KERNEL_SIZE):
#                 mask3 = ((y + j - mid) >= 0) & ((y + j - mid) < height)
#                 mask = (mask1 & mask2 & mask3)
#                 # compute masked indices
#                 sigma_masked = sigma[mask]
#                 x_masked = x[mask]
#                 y_masked = y[mask]
#                 dx = x + i - mid
#                 dy = y + j - mid
#                 dx_masked = dx[mask]
#                 dy_masked = dy[mask]

#                 # value at kernel i-mid, j-mid
#                 g = (1.0 / ((sigma_masked+1e-8)**2)) * torch.exp(-2.0 * ((i-mid)**2 + (j-mid)**2) / ((sigma_masked+1e-8)**2))
#                 kernel_sum[n,x_masked,y_masked] += g
#                 defocus_stack[n,x_masked,y_masked,0] += g * aif[dx_masked,dy_masked,0]
#                 defocus_stack[n,x_masked,y_masked,1] += g * aif[dx_masked,dy_masked,1]
#                 defocus_stack[n,x_masked,y_masked,2] += g * aif[dx_masked,dy_masked,2]

#         defocus_stack[n,x[mask1],y[mask1],0] /= kernel_sum[n,x[mask1],y[mask1]]
#         defocus_stack[n,x[mask1],y[mask1],1] /= kernel_sum[n,x[mask1],y[mask1]]
#         defocus_stack[n,x[mask1],y[mask1],2] /= kernel_sum[n,x[mask1],y[mask1]]

#         print((~mask).sum(), (mask1).sum())
#         defocus_stack[n,x[~mask1],y[~mask1],0] = aif[x[~mask1],y[~mask1],0]
#         defocus_stack[n,x[~mask1],y[~mask1],1] = aif[x[~mask1],y[~mask1],1]
#         defocus_stack[n,x[~mask1],y[~mask1],2] = aif[x[~mask1],y[~mask1],2]
    
#     return defocus_stack


# def forward_si_brute_force(dpt, aif):
    
#     r = computer(dpt, globals.Df)
#     # r = r.unsqueeze(-1).unsqueeze(-1)
    
#     width, height, fs = r.squeeze().shape

#     # y, x = torch.meshgrid(
#     #     torch.arange(width),
#     #     torch.arange(height),
#     #     indexing='ij')

#     # print(r.shape, y.shape, x.shape)

#     lim = globals.MAX_KERNEL_SIZE // 2
#     mid = globals.MAX_KERNEL_SIZE / 2
    
#     # G = torch.zeros((width, height, fs, globals.MAX_KERNEL_SIZE, globals.MAX_KERNEL_SIZE))
#     # G[:,:,:,lim,lim] = 1
#     kernel_sum = torch.zeros((fs, width, height))
#     defocus_stack = torch.zeros((fs, width, height, 3))
#     for n in range(fs):
#         defocus_stack[n] = aif.clone()

#     for x in range(width):
#         print(x,'/',width)
#         for y in range(height):
#             for n in range(fs):
#                 sigma = r[x,y,n]
#                 if (sigma > 1):
#                     for i in range(globals.MAX_KERNEL_SIZE):
#                         if ((x + i - lim) >= 0 and (x + i - lim) < width):
#                             for j in range(globals.MAX_KERNEL_SIZE):
#                                 if ((y + j - lim) >= 0 and (y + j - lim) < height):
#                                     g = (1.0 / (sigma * sigma)) * torch.exp(-2.0 * ((i-lim) * (i-lim) + (j-lim) * (j-lim)) / (sigma * sigma))
#                                     kernel_sum[n,x,y] += g
#                                     defocus_stack[n,x,y,0] += g * aif[x+i-lim,y+j-lim,0]
#                                     defocus_stack[n,x,y,1] += g * aif[x+i-lim,y+j-lim,0]
#                                     defocus_stack[n,x,y,2] += g * aif[x+i-lim,y+j-lim,0]
#                 else:
#                      kernel_sum[n,x,y] = 1

#     for n in range(fs):
#         defocus_stack = defocus_stack / kernel_sum.unsqueeze(-1)
    
#     return defocus_stack

# def computeG_si(r):
#     width, height, fs = r.squeeze().shape
    
#     x, y = torch.meshgrid(
#         torch.arange(width),
#         torch.arange(height),
#         indexing='ij')

#     print(r.shape, y.shape, x.shape)

#     mid = globals.MAX_KERNEL_SIZE // 2
#     # mid = globals.MAX_KERNEL_SIZE / 2
    
#     G = torch.zeros((width, height, fs, globals.MAX_KERNEL_SIZE, globals.MAX_KERNEL_SIZE))
#     G[:,:,:,mid,mid] = 1
    
#     for n in range(fs):
#         sigma = r[:,:,n].squeeze()
#         mask1 = sigma > 1
#         for i in range(globals.MAX_KERNEL_SIZE):
#             mask2 = ((x + i - mid) >= 0) & ((x + i - mid) < width)
#             for j in range(globals.MAX_KERNEL_SIZE):
#                 mask3 = ((y + j - mid) >= 0) & ((y + j - mid) < height)
#                 mask = (mask1 & mask2 & mask3)
#                 sigma_masked = sigma[mask]
#                 G[:,:,n,i,j][mask] = (1.0 / (sigma_masked * sigma_masked)) * torch.exp(-2.0 * ((i-mid) * (i-mid) + (j-mid) * (j-mid)) / (sigma_masked * sigma_masked))
    
#     norm = torch.sum(G, dim=(-2,-1)).unsqueeze(-1).unsqueeze(-1)
#     G /= norm

#     return G, norm

def computer(dpt, Df):
    Df_expanded = Df.view(1, 1, -1).to(dpt.device)
    # print(globals.f, globals.D, globals.ps)
    
    # # if we instead have sensor distance, must use
    # CoC = globals.D * globals.s * (1/globals.f - 1/(dpt.unsqueeze(-1)+1e-8) - 1/globals.s)

    # # si et al formulation -- is equivalent
    # sensor_dist = Df_expanded * globals.f / (Df_expanded - globals.f)
    # CoC = globals.D * sensor_dist * torch.abs(1/globals.f - 1/sensor_dist - 1/(dpt.unsqueeze(-1)+1e-8))        
    
    CoC = ((globals.D) 
        * (torch.abs(dpt.unsqueeze(-1) - Df_expanded) / (dpt.unsqueeze(-1)+1e-8)) 
        * (globals.f / (Df_expanded - globals.f)))
    
    # print('CoC',CoC.shape)
    # print(CoC)
    # for i in range(5):
    #     print('example pixel',CoC[237,110,i].item())
    # print('dpt:',dpt[237,110].item())
    
    r = CoC / 2. / globals.ps
    # for i in range(5):
    #     print('example sigma',r[237,110,i].item())
    
    r[torch.where(r < 2)] = 2
    # r[torch.where(r < 0.5)] = 0.5
    # r[torch.where(r < 1)] = 1
    # print('r.shape',r.shape)
    # TODO: si et al has 1 and it just doesnt do anything it seems, like fully ignores out of bounds
    
    
    return r

def computeG(r, u, v, kernel='gaussian'):
    assert kernel in ['gaussian', 'pillbox']
    # gaussian kernels
    # G = (1 / (2 * torch.pi * (r+1e-8)**2) * 
    #    torch.exp(-(u.to(r.device)**2 + v.to(r.device)**2) / (2 * (r+1e-8)**2)))

    # TODO: change this equation
    # the way they do it, they just IGNORE if r <= 1
    
    if kernel == 'gaussian':
        # todo 
        G = torch.exp(-(u.to(r.device)**2 + v.to(r.device)**2) / (2 * (r+1e-8)**2))
        # G = 1. / ((r+1e-8)**2) * torch.exp(-2.0 * (u.to(r.device)**2 + v.to(r.device)**2)/ ((r+1e-8)**2)) # si et al uses this formulation
    elif kernel == 'pillbox':
        # alternatively, use a pillbox
        G = ((u.to(r.device)**2 + v.to(r.device)**2) <= r**2).float()
    # print('G.shape',G.shape)

    # # ignore r <= 1
    # kernel = torch.zeros((globals.MAX_KERNEL_SIZE, globals.MAX_KERNEL_SIZE), dtype=G.dtype, device=G.device)
    # kernel[globals.MAX_KERNEL_SIZE // 2, globals.MAX_KERNEL_SIZE // 2] = 1.
    # mask3d = (r <= 2).squeeze(-1).squeeze(-1)
    # G[mask3d] = kernel

    # TODO: chekc this is working

    # for i in range(5):
    #     print('kernel:',G[237,110,i])

    # handle edge overflow -- THIS IS CAUSING ISSUES w/ COORD DESCENT
    # # print(globals.MAX_KERNEL_SIZE // 2)
    # if G.shape[0] > 1 and G.shape[1] > 1:
    #     # print('doing edge overflow')
    #     lim = globals.MAX_KERNEL_SIZE // 2
    #     for i in range(lim):
    #         if i < G.shape[0]:
    #             G[i,:,:,:(lim-i),:] = 0
    #         if i < G.shape[1]:
    #             G[:,i,:,:,:(lim-i)] = 0
    #         if (G.shape[0]-(i+1)) >= 0:
    #             G[(G.shape[0]-(i+1)),:,:,(lim+1+i):,:] = 0
    #         if (G.shape[1]-(i+1)) >= 0:
    #             G[:,(G.shape[1]-(i+1)),:,:,(lim+1+i):] = 0

    # print kernel at example point
    # print('kernel at 214, 54')
    # print(G[214,54,0])
    # print('r=',r[214,54,0])

    # print('after edges')
    # for i in range(5):
    #     print('kernel:',G[189+25,29+25,i])
        
    # # numerical stability remove low values 
    # G[G < 0.001] = 0

    # normalize the kernels
    norm = torch.sum(G, dim=(-2,-1)).unsqueeze(-1).unsqueeze(-1)
    # print('G nan?',torch.isnan(G).sum() > 0)
    G = G / (norm+1e-8)

    # print('(AFTER NORM) kernel at 214, 54')
    # print(G[214,54,0])
    
    # print('after norm')
    # for i in range(5):
    #     print('kernel:',G[237,110,i])
    return G, norm

def buildA(dpt, u, v, row, col, mask, use_torch=False, kernel='gaussian', op=None):
    # Find gaussian kernels from given dpt map 
    width, height = dpt.shape
    # print(width, height)
    r = computer(dpt, globals.Df)
    # print(r[189+25,29+25,:])
    # print('r',r[214,54])
    _, _, fs = r.shape
    r = r.unsqueeze(-1).unsqueeze(-1)
    
    
    if op == 'si':
        G, _ = computeG_si(r)
    else:
        G, _ = computeG(r, u, v, kernel=kernel)
    # print(G[101,87])

    A_stack = []
    for idx in range(fs):
        data = G[:,:,idx,:,:]
        data = data.flatten()
        data = data[mask]
        
        if use_torch:
            indices = torch.stack([row.to(dpt.device), col.to(dpt.device)])
            # A = torch.sparse_coo_tensor(torch.stack([row.to(dpt.device), col.to(dpt.device)]), data,
            #     (width*height, width*height), requires_grad=G.requires_grad)
            # A = A.coalesce().to_sparse_csr()
            A_stack.append((indices, data))
        else:
            A = scipy.sparse.csc_matrix((data, (row, col)),
                shape=(width*height, width*height))
            A_stack.append(A)

    return A_stack

def forward_single_pixel(i, j, Z, aif):#, max_kernel_size = 7):

    width, height, _ = aif.shape
    fs = len(globals.Df)


    dpt = torch.tensor([[Z]])

    
    r = computer(dpt, globals.Df)
    r = r.unsqueeze(-1).unsqueeze(-1)
    # print(r)

    u, v = compute_u_v()#max_kernel_size = max_kernel_size)
    # print(u, v)

    G, _ = computeG(r, u, v)
    # print(G)

    row = (i + u).flatten()
    col = (j + v).flatten()
    condition1 = row < 0
    condition2 = row >= width
    condition3 = col < 0
    condition4 = col >= height
    indices_to_delete = torch.where(condition1 | condition2 | condition3 | condition4)
    mask = torch.ones(col.size(0), dtype=torch.bool)
    mask[indices_to_delete] = False


    row = row[mask].to(int)
    col = col[mask].to(int)

    defocus_stack = torch.zeros((fs, 3)).to(aif.device)
    
    for idx in range(fs):
        G_idx = G[:,:,idx,:,:].flatten()[mask]

        aif_red = aif[row, col, 0]
        aif_green = aif[row, col, 1]
        aif_blue = aif[row, col, 2]

        defocus_stack[idx, 0] = (G_idx * aif_red).sum()
        defocus_stack[idx, 1] = (G_idx * aif_green).sum()
        defocus_stack[idx, 2] = (G_idx * aif_blue).sum()

    # for idx in range(fs):
    #     data = G[:,:,idx,:,:]
    #     aif = 
    
    # aif_red = aif[:,:,0].flatten().unsqueeze(1).to(dpt.device, dtype=dpt.dtype)
    # aif_green = aif[:,:,1].flatten().unsqueeze(1).to(dpt.device, dtype=dpt.dtype)
    # aif_blue = aif[:,:,2].flatten().unsqueeze(1).to(dpt.device, dtype=dpt.dtype)

    return defocus_stack

    
def forward_torch(dpt, aif, indices=None, kernel='gaussian', op=None):
    if indices is None:
        width, height = dpt.shape
        u, v, row, col, mask = precompute_indices(width, height)
    else:
        # print('here')
        u, v, row, col, mask = indices

    A_stack = buildA(dpt, u, v, row, col, mask, use_torch=True, kernel=kernel, op=op)
    width, height = dpt.shape

    defocus_stack = []

    # print('aif region')
    # lim = globals.MAX_KERNEL_SIZE //2
    # print('red')
    # print(aif[214-lim:214+lim+1,54-lim:54+lim+1,0])
    # print('green')
    # print(aif[214-lim:214+lim+1,54-lim:54+lim+1,1])
    # print('blue')
    # print(aif[214-lim:214+lim+1,54-lim:54+lim+1,2])


    aif_red = aif[:,:,0].flatten().unsqueeze(1).to(dpt.device, dtype=dpt.dtype)
    aif_green = aif[:,:,1].flatten().unsqueeze(1).to(dpt.device, dtype=dpt.dtype)
    aif_blue = aif[:,:,2].flatten().unsqueeze(1).to(dpt.device, dtype=dpt.dtype)

    # print('aif',aif[100,100])
    
    for idx in range(len(A_stack)):
        A_indices, A_values = A_stack[idx]

        b_red = torch_sparse.spmm(A_indices, A_values, 
            width*height, width*height, aif_red)
        b_green = torch_sparse.spmm(A_indices, A_values, 
            width*height, width*height, aif_green)
        b_blue = torch_sparse.spmm(A_indices, A_values, 
            width*height, width*height, aif_blue)
        
        b = torch.column_stack((b_red, b_green, b_blue))
        b = b.reshape((width,height,3))

        defocus_stack.append(b)

    return torch.stack(defocus_stack, dim=0)


def forward(dpt, aif, indices=None, kernel='gaussian'):
    if indices is None:
        width, height = dpt.shape
        u, v, row, col, mask = precompute_indices(width, height)
    else:
        u, v, row, col, mask = indices
    A_stack = buildA(dpt, u, v, row, col, mask, kernel=kernel)
    width, height = dpt.shape

    defocus_stack = []
    
    for idx in range(len(A_stack)):
        A = A_stack[idx]
        
        aif_red = aif[:,:,0].flatten()
        aif_green = aif[:,:,1].flatten()
        aif_blue = aif[:,:,2].flatten()
        b_red = A @ aif_red
        b_green = A @ aif_green
        b_blue = A @ aif_blue
        b = np.column_stack((b_red, b_green, b_blue))

        b = b.reshape((width,height,3))

        defocus_stack.append(b)
    return np.stack(defocus_stack, 0)
