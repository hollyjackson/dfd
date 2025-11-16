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

    us = np.linspace(-lim, lim, max_kernel_size, dtype=np.float32)
    vs = np.linspace(-lim, lim, max_kernel_size, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(us, vs, indexing='ij')
    u = grid_u[None, None, None, ...]
    v = grid_v[None, None, None, ...]

    return u, v

def compute_shifted_indices(width, height):
    max_kernel_size = globals.MAX_KERNEL_SIZE
    # lim = int((float(max_kernel_size) - 1) / 2.)
    lim = max_kernel_size // 2
    
    row_indices = np.zeros((width,height,max_kernel_size,max_kernel_size), dtype=np.intp)
    col_indices  = np.zeros((width,height,max_kernel_size,max_kernel_size), dtype=np.intp)
    
    grid = np.meshgrid(np.arange(width, dtype=np.intp), np.arange(height, dtype=np.intp), indexing='ij')
    indices = np.stack(grid, axis=-1)

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
    indices_to_delete = np.where(condition1 | condition2 | condition3 | condition4)
    mask = np.ones(col_indices.flatten().shape[0], dtype=bool)
    mask[indices_to_delete] = False

    return mask

def compute_mask_flattened_indices(row_indices, col_indices, mask, width, height):
    max_kernel_size = globals.MAX_KERNEL_SIZE
    
    flattened_indices = row_indices * (height) + col_indices
    
    row = np.arange(width*height, dtype=np.intp)
    row = np.expand_dims(row, 1)
    row = np.tile(row, (1, max_kernel_size*max_kernel_size))
    row = row.flatten()
    row = row[mask].astype(np.intp)
    
    col = flattened_indices.flatten()
    col = col[mask].astype(np.int32)

    return row, col

def precompute_indices(width, height):#, max_kernel_size = 7):

    print("precomputing indices")
    
    u, v = compute_u_v()#max_kernel_size = max_kernel_size)
    row_indices, col_indices = compute_shifted_indices(width, height)#, max_kernel_size = max_kernel_size)
    # grab indices to remove 
    mask = generate_mask(row_indices, col_indices, width, height)
    row, col = compute_mask_flattened_indices(row_indices, col_indices, mask, width, height)#, max_kernel_size = max_kernel_size)

    return u, v, row, col, mask


def computer(dpt, Df):

    # format focus setting
    if not isinstance(Df, np.ndarray):
        Df = Df.numpy().astype(np.float32)
    Df_expanded = Df.reshape(1, 1, -1)
    # compute CoC
    CoC = ((globals.D) 
        * (np.abs(dpt[...,None] - Df_expanded) / (dpt[...,None]+1e-8)) 
        * (globals.f / (Df_expanded - globals.f)))
    r = CoC / 2. / globals.ps

    # threshold
    # r[np.where(r < 2)] = 2 # was used for nyuv2
    r[np.where(r < globals.thresh)] = globals.thresh

    
    return r

def computeG_old(r, u, v):
    # compute Gaussian kernels
    G = np.exp(-(u**2 + v**2) / (2 * (r+1e-8)**2))        
    # G = ((u**2 + v**2) <= r**2).astype(np.float32) # disc 
    
    # normalize gaussian kernels
    norm = np.sum(G, axis=(-2,-1), keepdims=True, dtype=np.float32)
    G /= (norm+1e-8)

    return G, norm

def computeG(r, u, v, eps=1e-8):
    # exploit Gaussian separability for speed up
    u2 = (u[..., :, :1] ** 2).astype(np.float32)
    v2 = (v[..., :1, :] ** 2).astype(np.float32)
    inv2sigma2 = 1 / (2 * (r+eps)**2)
    
    # 1D Gaussians (broadcasted)
    Gu = np.exp(-u2 * inv2sigma2)
    Gv = np.exp(-v2 * inv2sigma2)
    
    # # normalize 1D, so the 2D outer will already be normalized
    # sum_u = Gu.sum(axis=-2, keepdims=True) + eps
    # sum_v = Gv.sum(axis=-1, keepdims=True) + eps
    # Gu /= sum_u
    # Gv /= sum_v
    
    # assemble full 2D kernel
    G = Gu * Gv 
    # norm = np.ones_like(sum_u, dtype=np.float32)

    # normalize gaussian kernels
    norm = np.sum(G, axis=(-2,-1), keepdims=True, dtype=np.float32)
    G /= (norm+1e-8)
    
    return G, norm
    
def build_fixed_pattern_csr(width, height, fs, row, col, data, dtype=np.float32):
    row = np.asarray(row, dtype=np.int32)
    col = np.asarray(col, dtype=np.int32)
    data = np.asarray(data, dtype=dtype)

    order = np.lexsort((col, row))
    row_sorted, col_sorted = row[order], col[order]

    # csr structure ensures it stays in original row, col order
    indptr = np.zeros(width*height+1, dtype=np.int32)
    np.add.at(indptr, row_sorted + 1, 1)
    np.cumsum(indptr, out=indptr)
    indices = col_sorted.astype(np.int32, copy=False)

    # build matrices
    A_stack = []
    for idx in range(fs):
        # A = scipy.sparse.csr_matrix((data, (row, col)),
        #         shape=(width*height, width*height), dtype=data.dtype)
        # A.sort_indices()
        A = scipy.sparse.csr_matrix((data, indices, indptr),
                shape=(width*height, width*height), dtype=data.dtype, copy=False)
        A_stack.append(A)
        
    return A_stack, order

def buildA(dpt, u, v, row, col, mask, template_A_stack=None):
    
    # Find gaussian kernels from given dpt map 
    width, height = dpt.shape
    r = computer(dpt, globals.Df)
    _, _, fs = r.shape
    r = r[...,None,None]
    
    G, _ = computeG(r, u, v)
    
    A_stack = []
    if template_A_stack is not None:
        A_stack_cache, order = template_A_stack
        
    for idx in range(fs):
        data = G[:,:,idx,:,:]
        data = data.flatten()
        data = data[mask]
        
        if template_A_stack is None:
            # warning -- this is > 3x slower
            # print(data.dtype)
            A = scipy.sparse.csr_matrix((data, (row, col)),
                shape=(width*height, width*height), dtype=data.dtype)
            # A.sort_indices()
            A_stack.append(A)
        else:
            A = A_stack_cache[idx].copy()
            A.data[:] = data[order]
            # print(data.dtype)
            A_stack.append(A)
            
    return A_stack

# def forward_single_pixel(i, j, Z, aif):#, max_kernel_size = 7):

#     width, height, _ = aif.shape
#     fs = len(globals.Df)

#     dpt = torch.tensor([[Z]])

#     r = computer(dpt, globals.Df)
#     r = r.unsqueeze(-1).unsqueeze(-1)

#     u, v = compute_u_v()#max_kernel_size = max_kernel_size)
#     # print(u, v)

#     G, _ = computeG(r, u, v)
#     # print(G)

#     row = (i + u).flatten()
#     col = (j + v).flatten()
#     condition1 = row < 0
#     condition2 = row >= width
#     condition3 = col < 0
#     condition4 = col >= height
#     indices_to_delete = torch.where(condition1 | condition2 | condition3 | condition4)
#     mask = torch.ones(col.size(0), dtype=torch.bool)
#     mask[indices_to_delete] = False


#     row = row[mask].to(int)
#     col = col[mask].to(int)

#     defocus_stack = torch.zeros((fs, 3)).to(aif.device)
    
#     for idx in range(fs):
#         G_idx = G[:,:,idx,:,:].flatten()[mask]

#         aif_red = aif[row, col, 0]
#         aif_green = aif[row, col, 1]
#         aif_blue = aif[row, col, 2]

#         defocus_stack[idx, 0] = (G_idx * aif_red).sum()
#         defocus_stack[idx, 1] = (G_idx * aif_green).sum()
#         defocus_stack[idx, 2] = (G_idx * aif_blue).sum()

#     # for idx in range(fs):
#     #     data = G[:,:,idx,:,:]
#     #     aif = 
    
#     # aif_red = aif[:,:,0].flatten().unsqueeze(1).to(dpt.device, dtype=dpt.dtype)
#     # aif_green = aif[:,:,1].flatten().unsqueeze(1).to(dpt.device, dtype=dpt.dtype)
#     # aif_blue = aif[:,:,2].flatten().unsqueeze(1).to(dpt.device, dtype=dpt.dtype)

#     return defocus_stack

    
# def forward_torch(dpt, aif, indices=None, kernel='gaussian', op=None, template_A_stack=None):
#     if indices is None:
#         width, height = dpt.shape
#         u, v, row, col, mask = precompute_indices(width, height)
#     else:
#         # print('here')
#         u, v, row, col, mask = indices

#     A_stack = buildA(dpt, u, v, row, col, mask, use_torch=True, kernel=kernel, op=op, template_A_stack=template_A_stack)
#     width, height = dpt.shape

#     defocus_stack = []

#     # print('aif region')
#     # lim = globals.MAX_KERNEL_SIZE //2
#     # print('red')
#     # print(aif[214-lim:214+lim+1,54-lim:54+lim+1,0])
#     # print('green')
#     # print(aif[214-lim:214+lim+1,54-lim:54+lim+1,1])
#     # print('blue')
#     # print(aif[214-lim:214+lim+1,54-lim:54+lim+1,2])


#     aif_red = aif[:,:,0].flatten().unsqueeze(1).to(dpt.device, dtype=dpt.dtype)
#     aif_green = aif[:,:,1].flatten().unsqueeze(1).to(dpt.device, dtype=dpt.dtype)
#     aif_blue = aif[:,:,2].flatten().unsqueeze(1).to(dpt.device, dtype=dpt.dtype)

#     # print('aif',aif[100,100])
    
#     for idx in range(len(A_stack)):
#         A_indices, A_values = A_stack[idx]

#         b_red = torch_sparse.spmm(A_indices, A_values, 
#             width*height, width*height, aif_red)
#         b_green = torch_sparse.spmm(A_indices, A_values, 
#             width*height, width*height, aif_green)
#         b_blue = torch_sparse.spmm(A_indices, A_values, 
#             width*height, width*height, aif_blue)
        
#         b = torch.column_stack((b_red, b_green, b_blue))
#         b = b.reshape((width,height,3))

#         defocus_stack.append(b)

#     return torch.stack(defocus_stack, dim=0)


def forward(dpt, aif, indices=None, template_A_stack=None):
    width, height = dpt.shape
    
    if indices is None:
        u, v, row, col, mask = precompute_indices(width, height)
    else:
        u, v, row, col, mask = indices
    
    A_stack = buildA(dpt, u, v, row, col, mask, template_A_stack=template_A_stack)
    
    defocus_stack = []

    aif_red = aif[:,:,0].flatten()
    aif_green = aif[:,:,1].flatten()
    aif_blue = aif[:,:,2].flatten()
    
    for idx in range(len(A_stack)):
        A = A_stack[idx]
        
        b_red = A @ aif_red
        b_green = A @ aif_green
        b_blue = A @ aif_blue
        b = np.column_stack((b_red, b_green, b_blue))

        b = b.reshape((width,height,3))

        defocus_stack.append(b)
    
    return np.stack(defocus_stack, 0)
