import os
import math
import sys

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
from globals import *

sys.path.append(os.path.join(os.getcwd(),'functions'))
from LBFGS import FullBatchLBFGS

import section_search


def save_progress(epoch, dpt_opt, losses, grad_norms=None, path=None):
    if path == None:
        path = os.getcwd()

    with torch.no_grad():
        plt.imshow(dpt_opt.cpu().numpy())
        plt.colorbar(label="depth (m)")
        plt.savefig(os.path.join(path,'dpt'+str(epoch)+'.png'))
        plt.close()
        # print(dpt_opt.cpu().numpy().min(),dpt_opt.cpu().numpy().max())
        
        plt.plot(losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(path,'loss.png'))
        plt.close()
        
        plt.plot([math.log(loss,10) for loss in losses])
        plt.xlabel('epoch')
        plt.ylabel('log(loss)')
        plt.savefig(os.path.join(path,'log_loss.png'))
        plt.close()

        if grad_norms != None:
            plt.plot(grad_norms)
            plt.xlabel('epoch')
            plt.ylabel('grad norm')
            plt.savefig(os.path.join(path,'grad_norm.png'))
            plt.close()
            
            plt.plot([math.log(g,10) for g in grad_norms])
            plt.xlabel('epoch')
            plt.ylabel('log(grad norm)')
            plt.savefig(os.path.join(path,'log_grad_norm.png'))
            plt.close()

# -------------------------------------


def grad_descent_scipy(forward, defocus_stack, dpt_initial, aif,
    method="L-BFGS-B", disp=True, maxiter=1000):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # defocus_stack = defocus_stack.to(device)
    

    width, height = dpt_initial.shape
    u, v, row, col, mask = forward_model.precompute_indices(width, height)
    indices = (u, v, row, col, mask)

    criterion = torch.nn.MSELoss()

    def loss_func(dpt):
        dpt_torch = torch.from_numpy(dpt).reshape(width,height).to(device)
        dpt_torch.requires_grad = True
        defocus_stack_pred = forward(dpt_torch, aif, indices)
        # loss_val = ((defocus_stack_pred - defocus_stack.to(device)) ** 2).mean()
        loss_val = criterion(defocus_stack_pred, defocus_stack.to(device))
        return loss_val.detach().cpu().item()
    
    def gradient(dpt):
        dpt_torch = torch.from_numpy(dpt).reshape(width,height).to(device)
        dpt_torch.requires_grad = True
        pred = forward(dpt_torch, aif, indices)
        loss_val = criterion(pred, defocus_stack.to(device))
        # compute gradient via autograd
        grad_val = torch.autograd.grad(loss_val, dpt_torch,
            create_graph=False, retain_graph=False)
        grad_val = grad_val[0]
        grad_val_np = grad_val.detach().cpu().numpy().flatten()

        # having memory issues
        del dpt_torch, pred, loss_val, grad_val
        torch.cuda.empty_cache()

        return grad_val_np

    dpt_opt = dpt_initial.flatten().cpu().numpy()#.to(device)
    print(dpt_opt.shape)
    
    res = scipy.optimize.minimize(loss_func, dpt_opt,
        jac=gradient, method=method,
        options={'disp': disp, 'maxiter': maxiter}) 

    print("Optimal value:", res.x)
    print("Function value at minimum:", res.fun)
    print("Converged:", res.success)

    dpt_opt = res.x.reshape((width,height))

    plt.imshow(dpt_opt)
    plt.colorbar(label="Intensity")
    plt.savefig('final.png')
    plt.close()

    return dpt_opt


# -------------------------------------


def diagonal_newton(forward, defocus_stack, dpt_initial, aif,
    disp=True, maxiter=1000, epsilon=1e-6, lr=1.0,
    path=None, plot_freq=None, print_freq=None,
    to_plot=True, to_print=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    defocus_stack = defocus_stack.to(device)
    
    width, height = dpt_initial.shape
    u, v, row, col, mask = forward_model.precompute_indices(width, height)
    indices = (u, v, row, col, mask)

    criterion = torch.nn.MSELoss()

    def loss_func(dpt):
        dpt_torch = torch.from_numpy(dpt).reshape(width,height).to(device)
        dpt_torch.requires_grad = True
    
        defocus_stack_pred = forward(dpt_torch, aif, indices)
        loss_val = criterion(defocus_stack_pred, defocus_stack.to(device))
    
        return loss_val.detach().cpu().item()

    def loss_func_torch(dpt_torch):
        defocus_stack_pred = forward(dpt_torch, aif, indices)
        return criterion(defocus_stack_pred, defocus_stack.to(device))#.item()
        
    def gradient(dpt):
        dpt_torch = torch.from_numpy(dpt).reshape(width,height).to(device)
        dpt_torch.requires_grad = True
    
        pred = forward(dpt_torch, aif, indices)
        loss_val = criterion(pred, defocus_stack.to(device))
    
        # compute gradient via autograd
        grad_val = torch.autograd.grad(loss_val, dpt_torch,
            create_graph=False, retain_graph=False)[0]
        # grad_val = grad_val[0]
        grad_val_np = grad_val.detach().cpu().numpy()#.flatten()

        # having memory issues
        del dpt_torch, pred, loss_val, grad_val
        torch.cuda.empty_cache()

        return grad_val_np

    def diagonal_hessian_vhp(dpt):
        pass

    # def diagonal_hessian_sparse(dpt):
    #     def compute_hessian_entry(func, x, i, j):
    #         g = grad(func(x), x, create_graph=True)[0]
    #         return grad(g[j], x, create_graph=True)[0][i]
    #     pass

    def diagonal_hessian_torch(dpt):
        dpt_torch = torch.from_numpy(dpt).reshape(width,height).to(device)
        dpt_torch.requires_grad = True
    
        # pred = forward(dpt_torch, aif, indices)
        # loss_val = criterion(pred, defocus_stack.to(device))
        print('computing hessian')
        H = torch.autograd.functional.hessian(loss_func_torch, dpt_torch)
        print(H.shape)

        H_diag = torch.einsum("iijj->ij", H)
        print(H_diag.shape)

        return H_diag.detach().cpu().numpy()

    def diagonal_hessian_parallel(dpt):
        # hessian is a 4d tensor of shape m x n by m x n
        # diagonal of the hessian is just m x n

        dpt_torch = torch.from_numpy(dpt).reshape(width,height).to(device)
        dpt_torch.requires_grad = True
    
        pred = forward(dpt_torch, aif, indices)
        loss_val = criterion(pred, defocus_stack.to(device))
    
        print('grad value')
        # compute gradient via autograd
        grad_val = torch.autograd.grad(loss_val, dpt_torch,
            create_graph=True, retain_graph=True)[0]
        # grad_val.requires_grad = True
    
        # compute hessian via autograd
        # h_diag = torch.zeros_like(dpt_torch)

        num_elements = dpt_torch.numel()
        # eye = torch.eye(num_elements, device=dpt_torch.device,
        #     dtype=dpt_torch.dtype).reshape(num_elements, *dpt_torch.shape)

        # Create a sparse identity matrix in vectorized form
        eye_indices = torch.arange(num_elements, device=dpt_torch.device).unsqueeze(0)  # (1, num_elements)
        eye_values = torch.ones(num_elements, device=dpt_torch.device)  # Non-zero values

        # Create sparse one-hot encoding matrix
        eye = torch.sparse_coo_tensor(eye_indices, eye_values,
            (num_elements,), dtype=dpt_torch.dtype)

        # Compute Hessian-vector products in parallel
        H_v = torch.autograd.grad(grad_val, dpt_torch,
            grad_outputs=eye, retain_graph=True)[0]

        # Extract only the diagonal
        H_diag = H_v.view(num_elements)[torch.arange(num_elements)].reshape_as(dpt_torch)

        del v, H_v
        torch.cuda.empty_cache()

        return H_diag.cpu().numpy()


    def diagonal_hessian(dpt):
        # hessian is a 4d tensor of shape m x n by m x n
        # diagonal of the hessian is just m x n

        dpt_torch = torch.from_numpy(dpt).reshape(width,height).to(device)
        dpt_torch.requires_grad = True
    
        pred = forward(dpt_torch, aif, indices)
        loss_val = criterion(pred, defocus_stack.to(device))
    
        
        
        # compute hessian via autograd
        h_diag = torch.zeros_like(dpt_torch)
        
        # print('grad value')
        # # compute gradient via autograd
        # grad_val = torch.autograd.grad(loss_val, dpt_torch,
        #     create_graph=True, retain_graph=True)[0]
        # for i in range(width):
        #     for j in range(height):
        #         print(i,j)
        #         g_ij = grad_val[i, j]
        #         h_ij, = torch.autograd.grad(g_ij, dpt_torch,
        #             retain_graph=True)
        #         h_diag[i,j] = h_ij[i,j].detach()
        # print('hessian computation')
        batch_size = dpt_torch.numel()
        for i in range(width * height):
            # print(i,'/',batch_size)
            v = torch.zeros_like(dpt_torch).view(-1)
            v[i] = 1 # single element perturbation
            v = v.view_as(dpt_torch)
            # compute Hessian vector product
            # Hv = torch.autograd.grad(grad_val, dpt_torch,
            #     grad_outputs=v, retain_graph=True)[0]
            _, Hv = torch.autograd.functional.vhp(loss_func_torch,
                dpt_torch, v)
            h_diag.view(-1)[i] = Hv.view(-1)[i]

        
        del dpt_torch, pred, loss_val, grad_val
        # del h_ij, g_ij
        del v, Hv

        return h_diag.cpu().numpy()

    dpt_opt = dpt_initial.cpu().numpy()
    print(dpt_opt.shape)

    if to_print:
        print("\n{:>6}  {:>12}  {:>12}".format("Epoch", "Loss", "|Proj g|"))
    
    losses = [loss_func(dpt_opt)]
    grad_norms = [np.linalg.norm(gradient(dpt_opt))]

    for epoch in range(maxiter):
        h_diag = diagonal_hessian(dpt_opt)
        grad = gradient(dpt_opt)

        h_diag_inv = 1.0 / (h_diag + epsilon)
        
        dpt_opt = dpt_opt - lr * (grad * h_diag_inv)

        loss = loss_func(dpt_opt)
        grad_norm = np.linalg.norm(grad)
        losses.append(loss)
        grad_norms.append(grad_norm)

        if loss < 1e-5:
            print('optimal after',epoch+1,'iterations')
            break

        if to_plot and epoch % plot_freq == 0:
            save_progress(epoch, dpt_opt, losses, grad_norms, path=path)
        if to_print and epoch % print_freq == 0:
            print("{:>6}  {:>12}  {:>12}".format(epoch, utils.format_number(loss), utils.format_number(grad_norm)))


    print('first loss:', losses[0], ', first grad norm:', utils.format_number(grad_norms[0]))
    print('final loss:', loss, ', final grad norm:', utils.format_number(grad_norm))
    print('range:', dpt_opt.min(), '-', dpt_opt.max())
    save_progress(epoch, dpt_opt, losses, grad_norms, path=path)


    plt.imshow(dpt_opt)
    plt.colorbar(label="Intensity")
    plt.savefig('final.png')
    plt.close()

    return dpt_opt


def grad_descent(forward, defocus_stack, dpt_initial, aif,
    method="Adam", num_epochs=1000, learning_rate=0.01,
    path=None, plot_freq=None, print_freq=None,
    to_plot=True, to_print=True):
    
    assert method in ["Adam", "LBFGS", "FullBatchLBFGS"]

    if plot_freq == None:
        if method in ["LBFGS", "FullBatchLBFGS"]:
            plot_freq = 10
        elif method == "Adam":
            plot_freq = 100
    if print_freq == None:
        print_freq = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dpt_opt = torch.nn.Parameter(dpt_initial.to(device))
    
    if method == "Adam":
        optimizer = torch.optim.Adam([dpt_opt], lr=learning_rate) # Adam
    elif method == "LBFGS":
        optimizer = torch.optim.LBFGS([dpt_opt],
            max_iter=100, line_search_fn="strong_wolfe") # LBFGS
    elif method == "FullBatchLBFGS":
        optimizer = FullBatchLBFGS([dpt_opt],
            line_search="Wolfe")

    criterion = torch.nn.MSELoss()
    defocus_stack = defocus_stack.to(device)

    width, height = dpt_initial.shape
    u, v, row, col, mask = forward_model.precompute_indices(width, height)
    indices = (u, v, row, col, mask)

    def closure():
        optimizer.zero_grad()  # Zero the gradients
        defocus_stack_pred = forward(dpt_opt, aif, indices)
        loss = criterion(defocus_stack_pred, defocus_stack)
        loss.backward()
        return loss
    def closure2():
        optimizer.zero_grad()  # Zero the gradients
        defocus_stack_pred = forward(dpt_opt, aif, indices)
        loss = criterion(defocus_stack_pred, defocus_stack)
        #loss.backward()
        return loss
        


    losses = []
    grad_norms = []
    if method == "FullBatchLBFGS":
        loss = closure2()
        loss.backward()
        losses.append(loss.item())
        grad_norm = torch.norm(dpt_opt.grad, p=2).item()
        grad_norms.append(grad_norm)

    if to_print:
        print("\n{:>6}  {:>12}  {:>12}".format("Epoch", "Loss", "|Proj g|"))
    for epoch in range(num_epochs):
        
        if method == "FullBatchLBFGS":
            options = {'closure': closure2, 'current_loss': loss,
                        'c1': 1e-4, 'c2':0.9, 'max_ls': 20}
            loss, _, t, ls_step, closure_eval, _, _, fail = optimizer.step(options=options)
            # print(loss.item(), t, ls_step, closure_eval, fail)
        elif method == "Adam" or method == "LBFGS":    
            optimizer.step(closure)
            loss = closure()

        losses.append(loss.item())
        grad_norm = torch.norm(dpt_opt.grad, p=2).item()
        grad_norms.append(grad_norm)

        if loss.item() < 1e-5:
            print('optimal after',epoch+1,'iterations')
            break
        # if loss.item() > losses[-2]:
        #     print('loss increasing,',epoch+1,'iterations')
        #     break
                
        with torch.no_grad():
            # plt.imsave(str(epoch)+'.png',dpt_opt.cpu().numpy(),vmin=0.7,vmax=1.9)
            if to_plot and epoch % plot_freq == 0:
                save_progress(epoch,dpt_opt,losses,grad_norms,path=path)
            if to_print and epoch % print_freq == 0:
                # print(f"Epoch {epoch+1}, Loss: {loss.item()}, |Proj g|: {torch.norm(dpt_opt.grad, p=2).item()}")
                print("{:>6}  {:>12}  {:>12}".format(epoch, utils.format_number(loss.item()), utils.format_number(grad_norm)))
    
    print('first loss:', losses[0], ', first grad norm:', utils.format_number(grad_norms[0]))
    print('final loss:', loss.item(), ', final grad norm:', utils.format_number(grad_norm))
    print('range:', dpt_opt.detach().cpu().min(), '-', dpt_opt.detach().cpu().max())
    save_progress(epoch,dpt_opt,losses,grad_norms,path=path)


    # if method == "FullBatchLBFGS":

    #     loss = closure2()
    #     loss.backward()
    #     losses = []
    #     for epoch in range(num_epochs):

    #         options = {'closure': closure2, 'current_loss': loss,
    #                 'c1': 1e-4, 'c2':0.9, 'max_ls': 20}
    #         loss, _, t, ls_step, closure_eval, _, _, fail = optimizer.step(options=options)
    #         print(loss.item(), t, ls_step, closure_eval, fail)
    #         losses.append(loss.item())
    #         # grad, obj = get_grad(optimizer, dpt_opt, defocus_stack, opfun)
    #         # p = optimizer.two_loop_recursion(-grad)
    #         # options = {'closure': closure2, 'current_loss':obj}
    # else:
    #     losses = []
    #     for epoch in range(num_epochs):

    #         optimizer.step(closure)
    #         loss = closure()
    #         losses.append(loss.item())

    #         if loss.item() < 1e-5:
    #             print(loss.item())
    #             print('optimal after',epoch,'iterations')
    #             save_progress(epoch,dpt_opt,losses)
    #             break
                    
    #         with torch.no_grad():
    #             if (method == "LBFGS" and epoch % 10 == 0) or (method == "Adam" and epoch % 100 == 0):
    #                 save_progress(epoch,dpt_opt,losses)
    #             if epoch % 10 == 0:
    #                 print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        
    return dpt_opt.detach().cpu()  # Return the optimized input tensor (final solution)

def grad_descent_with_linesearch(forward, defocus_stack, dpt_initial, aif,
    num_epochs=1000, c1=1e-4, c2=0.9, max_ls_iters=20, linesearch=True):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dpt_opt = torch.nn.Parameter(dpt_initial.to(device))
    dpt_opt.requires_grad_(True)
    
    criterion = torch.nn.MSELoss()
    defocus_stack = defocus_stack.to(device)

    width, height = dpt_initial.shape
    u, v, row, col, mask = forward_model.precompute_indices(width, height)
    indices = (u, v, row, col, mask)

    losses = []
    
    # helper fxn to compute f(x) and grad f(x)
    def f_and_grad(x):
        pred = forward(x, aif, indices)
        loss_val = criterion(pred, defocus_stack)
        # compute gradient via autograd
        # TODO: check to make sure that this is grabbing the correct gradient -- what is the shape of the output?
        grad_val = torch.autograd.grad(loss_val, x,
            create_graph=False, retain_graph=False)
        # print("shape of grad_val", grad_val[0].shape)
        grad_val = grad_val[0]
        return loss_val, grad_val

    # # fixed step length!! -- for experimenting
    # t = 0.5
    t = 1.0#0.01
    
    for epoch in range(num_epochs):

        # ------------------------------------------------------
        # compute f(x) and grad f(x) at the current x (dpt map)
        loss_val, grad_val = f_and_grad(dpt_opt)
        losses.append(loss_val.item())
        # ------------------------------------------------------

        if loss_val.item() < 1e-5:
            print(loss_val.item())
            print('optimal after',epoch,'iterations')
            break

        # direction is the negative gradient (steepest descent)
        direction = -grad_val

        # f'(x; d) = -||grad(f(x))||^2
        # f_prime_x_d = -torch.norm(grad_val, 2)
        f_prime_x_d = grad_val.flatten().T @ direction.flatten()

        if linesearch:
            # print('here')
            # initialize bracketing parameters
            alpha = 0.0
            # t = 1.0  # initial step size guess
            beta = float('inf')

            f_x = loss_val.item()  # f(x)
            
            for ls_iter in range(max_ls_iters):
                # evaluate f(x + t d) and grad f(x + t d)
                x_temp = dpt_opt + t * direction
                f_x_t, grad_x_t = f_and_grad(x_temp)

                # armijo check: f(x + t d) > f(x) + c1 * t * f'(x; d)
                if f_x_t > f_x + c1 * t * f_prime_x_d:
                    beta = t
                    t = 0.5 * (alpha + beta)
                # curvature check: f'(x + t d) < c2 * f'(x; d)
                else:
                    # f'(x+t*d;d) = -grad(f(x-t*gradf(x))^T grad(f(x))
                    # f_prime_xtd_d = -grad_x_t.T.flatten() @ grad_val.flatten()
                    f_prime_xtd_d = grad_x_t.flatten().T @ direction.flatten()
                    # print("f_prime_xtd_d",f_prime_xtd_d.shape,f_prime_xtd_d.item())
                    if f_prime_xtd_d.item() < c2 * f_prime_x_d:
                        alpha = t
                        if beta == float('inf'): # expand outward
                            t = 2.0 * alpha
                        else: # bisect the bracket
                            t = 0.5 * (alpha + beta)
                    else:
                        break

        # UPDATE STEP
        with torch.no_grad():
            dpt_opt += t * direction
            dpt_opt.requires_grad_(True)  # re-enable grad for next iteration

            if (epoch + 1) % 100 == 0:
                save_progress(epoch,dpt_opt,losses)

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss_val.item()}")

    return dpt_opt.detach().cpu()

if __name__ == "__main__":

    aif, dpt, gt_defocus_stack = utils.load_single_sample(fs=5)
    defocus_stack_torch = forward_model.forward_torch(dpt, aif)#.float() / 255.0)
    
    # initializations

    # gray = cv2.cvtColor(aif.numpy(), cv2.COLOR_BGR2GRAY).astype(np.float32)
    # noise = np.random.normal(0., 25., gray.shape).astype(np.float32)
    # print(gray.dtype, noise.dtype)
    # dpt_initial = cv2.add(gray, noise)
    # # rescale to be between 0.1 and 10
    # dpt_initial = (dpt_initial - np.min(dpt_initial)) * (10 - 0.1) / (np.max(dpt_initial) - np.min(dpt_initial)) + 0.1
    # dpt_initial = torch.tensor(dpt_initial)
    
    # constant = torch.full_like(dpt, 1)
    # noise = 0.1 * torch.randn_like(dpt)
    # dpt_initial = constant + noise
    # print(torch.min(dpt_initial),torch.max(dpt_initial))
    # dpt_initial = dpt

    # experiment -- start at true depth map and add tiny amount of noise
    # noise = 0.1 * torch.randn_like(dpt)
    # dpt_initial = dpt + noise

    dpt_initial, _, _ = section_search.grid_search(aif, defocus_stack_torch)
    dpt_initial = torch.from_numpy(dpt_initial.astype(np.float32))
    print(dpt_initial.dtype, dpt.dtype)
    dpt_initial = dpt
    # dpt_initial = torch.clip(dpt_initial, min=0.1, max=10.0)
    # print(torch.sum(dpt_initial < 0))

    
    # plt.imsave('init.png',dpt_initial, vmin=0.7, vmax=1.9)

    plt.imshow(dpt_initial, vmin=0.7, vmax=1.9)
    plt.colorbar(label="Intensity")
    plt.title('dpt -- initialization')
    plt.show()

    plt.imshow(dpt)
    plt.colorbar(label="Intensity")
    plt.title('dpt -- ground truth')
    plt.show()
    
    # forward model (torch)
    utils.plot_stacks_side_by_side(gt_defocus_stack, defocus_stack_torch, globals.Df)
    # plt.show()

    # GRADIENT DESCENT
    # dpt_recon = grad_descent_with_linesearch(forward_model.forward_torch, defocus_stack_torch,
    #     dpt_initial, aif.float())# / 255.0)
    # dpt_recon = grad_descent(forward_model.forward_torch, defocus_stack_torch,
    #     dpt_initial, aif.float(), method="Adam")# / 255.0)
    # dpt_recon = grad_descent_scipy(forward_model.forward_torch,
    #     defocus_stack_torch, dpt_initial, aif)
    dpt_recon = diagonal_newton(forward_model.forward_torch,
        defocus_stack_torch, dpt_initial, aif, print_freq=1)

    utils.plot_compare_greyscale(dpt_recon, dpt, vmin=0.7, vmax=1.9)
    plt.show()

    diff = torch.abs(dpt_recon - dpt)
    plt.imshow(diff.numpy())
    plt.title('diff')
    plt.show()



    num_worst_pixels = 5
    worst_indices = torch.topk(diff.view(-1), num_worst_pixels).indices
    worst_coords = [(idx // diff.shape[1], idx % diff.shape[1]) for idx in worst_indices]

    plt.imshow(dpt_recon.numpy())
    plt.scatter([y.item() for x, y in worst_coords], [x.item() for x, y in worst_coords], color='red', marker='x', s=100, label='Worst Diff Pixels')
    plt.title('Worst Difference Pixels Over Image')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    plt.legend()
    plt.show()

    print(worst_coords)

