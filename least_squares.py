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

    

def least_squares(dpt, defocus_stack, maxiter = 500):
    print('Least squares...')

    width, height = dpt.shape

    u, v, row, col, mask = forward_model.precompute_indices(width, height)
    A_stack = forward_model.buildA(dpt, u, v, row, col, mask)
    b_red_stack, b_green_stack, b_blue_stack = buildb(defocus_stack)
    
    A = scipy.sparse.vstack(A_stack)
    b_red = np.concatenate(b_red_stack)
    b_green = np.concatenate(b_green_stack)
    b_blue = np.concatenate(b_blue_stack)

    # def matvec(vec):
    #     return A.dot(vec)
    # def rmatvec(vec):
    #     return (A.getH()).dot(vec)
    # # create the LinearOperator for lsqr_linear
    # linear_op = scipy.sparse.linalg.LinearOperator(A.shape, matvec=matvec, rmatvec=rmatvec)

    # plt.figure()
    # plt.spy(A, markersize=1)
    # plt.title("Sparsity Pattern")
    # plt.xlabel("Columns")
    # plt.ylabel("Rows")
    # plt.show()

    # print(A.shape,b_red.shape)
    print("\n{:>7}  {:>12}  {:>12}  {:>6}".format("Channel", "r1norm", "norm(x)", "Num. Iters."))

    result = scipy.sparse.linalg.lsqr(A, b_red, iter_lim=maxiter)
    x_red = result[0]
    nit = result[2]
    residuals = result[3]
    
    # result = scipy.optimize.lsq_linear(linear_op, b_red, bounds=(0,255), verbose=2)#, show=True)#, iter_lim=500)
    # x_red = result[0]
    # residuals = result[2]
    # nit = result[6]
    
    # result = scipy.sparse.linalg.cg(A.T @ A, A.T @ b_red, maxiter=500)
    
    # output the results
    print("{:>7}  {:>12}  {:>12}  {:>6}".format(
        "red",
        utils.format_number(residuals),
        utils.format_number(np.linalg.norm(x_red)),
        nit
    ))
    
    result = scipy.sparse.linalg.lsqr(A, b_green, iter_lim=maxiter)
    x_green = result[0]
    nit = result[2]
    residuals = result[3]
    # result = scipy.optimize.lsq_linear(linear_op, b_green, bounds=(0,255), verbose=2)
    # x_green = result[0]
    # residuals = result[2]
    # nit = result[6]
    
    
    # output the results
    print("{:>7}  {:>12}  {:>12}  {:>6}".format(
        "green",
        utils.format_number(residuals),
        utils.format_number(np.linalg.norm(x_green)),
        nit
    ))

    result = scipy.sparse.linalg.lsqr(A, b_blue, iter_lim=maxiter)
    x_blue = result[0]
    nit = result[2]
    residuals = result[3]
    # result = scipy.optimize.lsq_linear(linear_op, b_blue, bounds=(0,255), verbose=2)
    # x_blue = result[0]
    # residuals = result[2]
    # nit = result[6]
    
    
    # output the results
    print("{:>7}  {:>12}  {:>12}  {:>6}".format(
        "blue",
        utils.format_number(residuals),
        utils.format_number(np.linalg.norm(x_blue)),
        nit
    ))



    recon_aif = np.column_stack((x_red, x_green, x_blue))
    # recon_aif = recon_aif#.astype(int)
    recon_aif = recon_aif.reshape((width,height,3))
        

    return recon_aif
