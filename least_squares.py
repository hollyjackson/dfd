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




def bounded_fista_3d(dpt, defocus_stack, IMAGE_RANGE, tol = 1e-6, maxiter = 1000, gt=None):
    
    print('Bounded FISTA...')

    width, height = dpt.shape

    u, v, row, col, mask = forward_model.precompute_indices(width, height)
    A_stack = forward_model.buildA(dpt, u, v, row, col, mask)
    b_red_stack, b_green_stack, b_blue_stack = buildb(defocus_stack)

    A = scipy.sparse.vstack(A_stack).tocsr()
    # print(A.shape,  A.count_nonzero() )
    A_T = A.T.tocsr()
    b_red = np.concatenate(b_red_stack)
    b_green = np.concatenate(b_green_stack)
    b_blue = np.concatenate(b_blue_stack)
    b = np.stack([b_red, b_green, b_blue], axis=1)

    # step size -- inverse of the Lipschitz constant of the gradient
    t0 = time.time()
    norm = scipy.sparse.linalg.norm(A, ord=2)
    eta = 1.0 / norm ** 2
    t1 = time.time()
    print('step size', t1-t0)
    
    aif = np.zeros((width*height, 3))
    aif_guess = aif.copy()
    Ay = A.dot(aif_guess)

    t = 1.0
        
    progress = tqdm.trange(maxiter, desc="Optimizing", leave=True)        
    for i in progress:#range(maxiter):
        # grad = A.T.dot(A.dot(aif) - b)
        r = Ay - b
        grad = A_T.dot(r)

        # fixed step size
        aif_new = aif_guess - eta * grad
        aif_new = np.clip(aif_new, 0, IMAGE_RANGE) # euclidean projection

        # momentum update
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        aif_guess = aif_new + ((t-1) / t_new) * (aif_new - aif)
        
        # # print(f(A, b, aif))
        # if i % 10 == 0:
        #     progress.set_postfix(loss=0.5*np.linalg.norm(r)**2, refresh=False)
        #     # compute mse w defocus stack
        #     if gt is not None:
        #         mse = np.mean((gt.numpy() - aif_new.reshape((width, height, 3)))**2)
        #         progress.set_postfix(mse=mse, refresh=False)
            
        if np.linalg.norm(aif_new - aif) < tol:
            print('Achieved tolerance')
            break

        aif = aif_new
        t = t_new
        Ay = A.dot(aif_guess)

    print('r1norm', np.linalg.norm(r), 'norm(x)', np.linalg.norm(aif))
    
    return aif.reshape((width, height, 3))


def bounded_projected_gradient_descent(dpt, defocus_stack, IMAGE_RANGE, eta0 = 1.0, alpha = 1e-4, beta = 0.5, tol = 1e-6, maxiter = 1000):
    
    def f(Ax, b):
        return 0.5 * np.linalg.norm(Ax - b)**2
        
    print('Bounded projected gradient descent...')

    width, height = dpt.shape

    u, v, row, col, mask = forward_model.precompute_indices(width, height)
    A_stack = forward_model.buildA(dpt, u, v, row, col, mask)
    b_red_stack, b_green_stack, b_blue_stack = buildb(defocus_stack)

    A = scipy.sparse.vstack(A_stack).tocsr()
    A_T = A.T.tocsr()
    b_red = np.concatenate(b_red_stack)
    b_green = np.concatenate(b_green_stack)
    b_blue = np.concatenate(b_blue_stack)


    # minimize squares Euclidean norm residual 
    # 1/2 ||Ax - b||_2^2
    # s.t. x \in [0, 255]

    # \nabla f (x) = A^T (Ax - b)

    eta = 0.5
    # eta = 1.0 / scipy.sparse.linalg.norm(A, ord=2) ** 2
    # print('step size', eta)

    recon_aif = []
    
    for b in [b_red, b_green, b_blue]:
        print('Projected gradient descent')
    
        aif = np.zeros((width*height))
        Ax = A.dot(aif)
        
        progress = tqdm.trange(maxiter, desc="Optimizing", leave=True)        
        for i in progress:#range(maxiter):
            # grad = A.T.dot(A.dot(aif) - b)
            # t0 = time.time()
            r = Ax - b
            grad = A_T.dot(r)
            # t1 = time.time()
            
            grad_norm_sq = np.linalg.norm(grad)**2
            # t2 = time.time()
            
            # # Armijo line search
            f_aif = 0.5 * np.linalg.norm(r)**2
            # t3 = time.time()
            # eta = eta0
            # while True:
            #     aif_new = aif - eta * grad
            #     aif_new = np.clip(aif_new, 0, IMAGE_RANGE) # euclidean projection
            #     Ax_new = A.dot(aif_new)
            #     if f(Ax_new, b) <= f_aif - alpha * eta * grad_norm_sq:
            #         break
            #     eta *= beta
            # print(eta)
            # t4 = time.time()

            # fixed step size
            aif_new = aif - eta * grad
            aif_new = np.clip(aif_new, 0, IMAGE_RANGE) # euclidean projection
            Ax_new = A.dot(aif_new)

            # print(f(A, b, aif))
            if i % 10 == 0:
                progress.set_postfix(loss=f_aif, refresh=False)
            
            if np.linalg.norm(aif_new - aif) < tol:
                print('Achieved tolerance')
                break

            # print(f"[{i}] grad: {t1 - t0:.3f}s | grand_norm_sq: {t2 - t1:.3f}s | f_aif: {t3 - t2:.3f}s | armijo: {t4 - t3:.3f}s | total: {t4 - t0:.3f}s")

    
            aif = aif_new
            Ax = Ax_new

        recon_aif.append(aif)
    
    recon_aif = np.column_stack(recon_aif).reshape((width, height, 3))

    return recon_aif

    
    

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
