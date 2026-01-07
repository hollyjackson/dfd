import os
import math

import numpy as np
import scipy
import matplotlib.pyplot as plt

from PIL import Image
import cv2
import tqdm
import time

import torch
torch.cuda.empty_cache()
import torch_sparse

import utils
import forward_model

def buildb(defocus_stack):
    b_red_stack = []
    b_green_stack = []
    b_blue_stack = []
    for idx in range(len(defocus_stack)):
        # print(idx)
        
        b_red = defocus_stack[idx][:,:,0].flatten()
        b_green = defocus_stack[idx][:,:,1].flatten()        
        b_blue = defocus_stack[idx][:,:,2].flatten()

        b_red_stack.append(b_red)
        b_green_stack.append(b_green)
        b_blue_stack.append(b_blue)
    return b_red_stack, b_green_stack, b_blue_stack     

def compute_Lipschitz_constant(A):
    norm = scipy.sparse.linalg.norm(A, ord=2)
    return norm**2

def approx_Lipschitz_constant(A, A_T, iters=15):
    # approximate by power iterations
    n = A.shape[1]
    x = np.random.standard_normal(n).astype(np.float32, copy=False)
    x /= (np.linalg.norm(x) + 1e-8)
    
    for _ in range(iters):
        y = A.dot(x)
        z = A_T.dot(y)
        nz = np.linalg.norm(z)
        if nz == 0:
            return 1.0
        x = z / nz
    y = A.dot(x)
    return np.float32(np.dot(y, y)) # Rayleight quotiet

def bounded_fista_3d(dpt, defocus_stack, IMAGE_RANGE, indices=None, template_A_stack=None, tol = 1e-6, maxiter = 1000, gt=None, verbose=True):

    if verbose:
        print('Bounded FISTA...')

    width, height = dpt.shape
    if indices is None:
        u, v, row, col, mask = forward_model.precompute_indices(width, height)
    else:
        u, v, row, col, mask = indices

    # t0 = time.time()
    A_stack = forward_model.buildA(dpt, u, v, row, col, mask, template_A_stack=template_A_stack)
    # print('A building', time.time()-t0)
    # t0 = time.time()
    b_red_stack, b_green_stack, b_blue_stack = buildb(defocus_stack)
    # print('b building', time.time()-t0)

    # t0 = time.time()
    A = scipy.sparse.vstack(A_stack).tocsr(copy=False)
    A.sort_indices() # seems to help make matrix multiplictaion a bit faster per iter but costs about 0.2 sec 
    assert A.dtype == np.float32
    # print('A stack', time.time()-t0)

    t0 = time.time()
    A_T = A.T#tocsc().transpose(copy=False).astype(np.float32, copy=False)
    # print('A_T', time.time()-t0)

    
    # t0 = time.time()
    b_red = np.concatenate(b_red_stack)
    b_green = np.concatenate(b_green_stack)
    b_blue = np.concatenate(b_blue_stack)
    b = np.stack([b_red, b_green, b_blue], axis=1).astype(np.float32, copy=False)
    # print('b stack', time.time() - t0)
    
    # step size -- inverse of the Lipschitz constant of the gradient
    # t0 = time.time()
    # # norm = scipy.sparse.linalg.norm(A, ord=2)
    # # eta = 1.0 / norm ** 2
    # L = compute_Lipschitz_constant(A)
    # eta = 1.0 / L
    # print('step size computation', time.time()-t0, eta)

    # t0 = time.time()
    L = approx_Lipschitz_constant(A, A_T)
    eta = 1.0 / L
    # print('step size computation', time.time()-t0, eta)
    
    aif = np.zeros((width*height, 3), dtype=np.float32)
    aif_guess = aif.copy()
    
    # Ay = A.dot(aif_guess)
    # t0 = time.time()
    Ay0 = A.dot(aif_guess[:, 0]); Ay1 = A.dot(aif_guess[:, 1]); Ay2 = A.dot(aif_guess[:, 2])
    Ay = np.column_stack((Ay0, Ay1, Ay2))
    # print('Ay all at once', time.time()-t0)
    
    
    t = 1.0
        
    progress = tqdm.trange(maxiter, desc="Optimizing", leave=True, disable=(not verbose))        
    for i in progress:#range(maxiter):
        # grad = A.T.dot(A.dot(aif) - b)
        # t0 = time.time()
        r = Ay - b
        # print('build r', time.time()-t0)
        # t0 = time.time()
        # grad = A_T.dot(r)
        g0 = A_T.dot(r[:, 0]); g1 = A_T.dot(r[:, 1]); g2 = A_T.dot(r[:, 2])
        grad = np.column_stack((g0, g1, g2))
        # print('grad', time.time() - t0)

        # fixed step size
        # t0 = time.time()
        aif_new = aif_guess - eta * grad
        # print('aif new', time.time() - t0)
        # t0 = time.time()
        aif_new = np.clip(aif_new, 0, IMAGE_RANGE) # euclidean projection
        # print('clip', time.time() - t0)

        # momentum update
        # t0 = time.time()
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        # print('t new', time.time() - t0)
        # t0 = time.time()
        aif_guess = aif_new + ((t-1) / t_new) * (aif_new - aif)
        # print('aif guess', time.time() - t0)
        
        # # print(f(A, b, aif))
        # if i % 10 == 0:
        #     progress.set_postfix(loss=0.5*np.linalg.norm(r)**2, refresh=False)
        #     # compute mse w defocus stack
        #     if gt is not None:
        #         mse = np.mean((gt.numpy() - aif_new.reshape((width, height, 3)))**2)
        #         progress.set_postfix(mse=mse, refresh=False)

        # t0 = time.time()
        aif_norm = np.linalg.norm(aif_new - aif)
        # print('aif norm', time.time() - t0)
        if aif_norm < tol:
            if verbose:
                print('Achieved tolerance')
            break

        aif = aif_new
        t = t_new
        # t0 = time.time()
        # Ay = A.dot(aif_guess)
        Ay0 = A.dot(aif_guess[:, 0]); Ay1 = A.dot(aif_guess[:, 1]); Ay2 = A.dot(aif_guess[:, 2])
        Ay = np.column_stack((Ay0, Ay1, Ay2))
        # print('new Ay', time.time()-t0)

    if verbose:
        print('r1norm', np.linalg.norm(r), 'norm(x)', np.linalg.norm(aif))
    
    return aif.reshape((width, height, 3))

