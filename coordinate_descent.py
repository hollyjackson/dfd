import os
import numpy as np
import matplotlib.pyplot as plt

import utils
import forward_model
import globals
import gradient_descent
import least_squares
import section_search
import torch

import cv2 as cv
import math

try:
    from skimage.restoration import denoise_tv_bregman, denoise_tv_chambolle
except ImportError:
    # skimage < 0.12
    from skimage.filters import denoise_tv_bregman, denoise_tv_chambolle

# def isotropic_tv_derivative(image, epsilon=1e-8):
#     m, n = image.shape
#     grad = np.zeros_like(image)
#     dx = np.zeros_like(image)
#     dy = np.zeros_like(image)
    
#     dx[:, :-1] = image[:, 1:] - image[:, :-1]
#     dy[:-1, :] = image[1:, :] - image[:-1, :]
#     magnitude = np.sqrt(dx**2 + dy**2 + epsilon)

#     grad[:, :-1] += dx[:, :-1] / magnitude[:, :-1]
#     grad[:-1, :] += dy[:-1, :] / magnitude[:-1, :]
#     grad[:, 1:] -= dx[:, :-1] / magnitude[:, :-1]
#     grad[1:, :] -= dy[:-1, :] / magnitude[:-1, :]
#     return grad
    
# def total_variation_torch(image):
#     tv_x = (image[:, 1:] - image[:, :-1])**2
#     tv_y = (image[1:, :] - image[:-1, :])**2
#     return torch.sqrt(tv_x + tv_y).sum()

# def total_variation(image):
#     tv_x = (image[:, 1:] - image[:, :-1])**2
#     tv_y = (image[1:, :] - image[:-1, :])**2
#     return np.sum(np.sqrt(tv_x + tv_y))



def coordinate_descent(defocus_stack,  gss_tol = 1e-2, gss_window = 1, ls_maxiter = 100, ls_maxiter_multiplier = None, num_epochs = 25, least_squares_first = True, save_plots = True, show_plots = False, depth_init = None, aif_init = None, dpt_denoising_weight = None, aif_denoising_weight = None, dpt_denoise_delay = 10, experiment_name = 'coord-descent', vmin = 0.7, vmax = 1.9, proxy_opt = False, beta = 1e-3, multiplier = 1.1, remove_outliers = False, diff_thresh = 2, tv_thresh = 0.15, tv_thresh_min = 0.15, tv_thresh_multiplier = None, outlier_patch_type = 'tv', adaptive_grid = False, grid_window = 0.25, gamma = 1e-3, similarity_penalty = False, finite_differences = False, t = None, fd_maxiter = 100, epsilon = 1e-3, min_Z = 0.1, max_Z = 10, num_Z = 100, use_CUDA = True, aif_method='fista'):
    assert not (finite_differences and adaptive_grid)
    assert not (finite_differences and similarity_penalty)
    assert aif_method in ['fista', 'ls']

    # important initializations ---------------------
    if save_plots:
        EXPERIMENT_NAME = experiment_name
        experiment_folder = utils.create_experiment_folder(EXPERIMENT_NAME)
    
    device = torch.device("cuda" if (torch.cuda.is_available() and use_CUDA) else "cpu")
    defocus_stack_torch = torch.from_numpy(defocus_stack).to(device)

    criterion = torch.nn.MSELoss()
    losses = []
    
    width = defocus_stack.shape[1]
    height = defocus_stack.shape[2]
    IMAGE_RANGE = 255. # if defocus stack in [0-255]
    if defocus_stack.max() <= 1.5: # instead in [0-1]
        IMAGE_RANGE = 1.
        print('Images in range [0-1]')
    else:
        print('Images in range [0-255]')
    # ------------------------------------------------
    
    def generate_AIF(aif, dpt, dpt_proxy, ls_maxiter, iter_folder=None):
        # --------------
        # least squares
        # --------------
        if aif_method == 'ls':
            aif = least_squares.least_squares(dpt.cpu(), defocus_stack, maxiter=ls_maxiter)
        else:
            aif = least_squares.bounded_fista_3d(dpt.cpu(), defocus_stack, IMAGE_RANGE, maxiter=ls_maxiter)
        print('\nAIF result range: [',aif.min(), ',', aif.max(),']')
        
        loss = criterion(forward_model.forward_torch(dpt, torch.from_numpy(aif)), defocus_stack_torch)
        losses.append(loss.item())
        print('Loss:',loss.item(), ', TV:',section_search.total_variation(aif))

        if aif_method == 'ls':
            aif = np.clip(aif, 0, IMAGE_RANGE) # TODO: edit this depending on range used 
        # aif = np.clip(aif, 0, 1)
        
        
        loss = criterion(forward_model.forward_torch(dpt, torch.from_numpy(aif)), defocus_stack_torch)
        losses.append(loss.item())
        print('Loss after clipping:',loss.item(),', TV:',section_search.total_variation(aif))
        print()

        if save_plots or show_plots:
            plt.imshow(aif / IMAGE_RANGE)
            if save_plots:
                plt.savefig(os.path.join(iter_folder,'least_squares_'+str(i)+'.png'))
            if show_plots:
                plt.show()
            else:
                plt.close()

        # ---------------------
        # TV denoising
        # ------------------
        if aif_denoising_weight != None:
            print('TV denoising')
            aif = denoise_tv_bregman(aif, weight=aif_denoising_weight)
            if save_plots:
                plt.imshow(aif / IMAGE_RANGE)
                plt.savefig(os.path.join(iter_folder,'aif_denoise_'+str(i)+'.png'))
                plt.close()
    
            loss = criterion(forward_model.forward_torch(dpt, torch.from_numpy(aif)), defocus_stack_torch)
            losses.append(loss.item())
            print('Loss after denoising:',loss.item(),', TV:',section_search.total_variation(aif))
            print()

        # -----------------------
        # TV denoising of proxy
        # -----------------------
        if proxy_opt:
            dpt_proxy = torch.from_numpy(denoise_tv_bregman(dpt.cpu().numpy(), weight=dpt_denoising_weight)).to(aif.device)

            if save_plots or show_plots:
                plt.imshow(dpt_proxy.cpu().numpy(), vmin=vmin, vmax=vmax)
                plt.colorbar()
                if save_plots:
                    plt.savefig(os.path.join(iter_folder,'dpt_proxy_'+str(i)+'.png'))
                if show_plots:
                    plt.show()
                else:
                    plt.close()

        return aif, dpt, dpt_proxy

    def generate_DPT(aif, dpt, dpt_proxy, beta=None, iter_folder=None, last_dpt=None):
        # ---------------------------
        # grid + backtracking search
        # ---------------------------
        aif = torch.from_numpy(aif).to(dpt.device, dtype=dpt.dtype)

        if finite_differences:
            depth_map_golden = section_search.finite_difference(aif, defocus_stack_torch, dpt, t_initial=t, max_iter=fd_maxiter, epsilon=epsilon, save_plots=save_plots, show_plots=show_plots, iter_folder=iter_folder, vmin=vmin, vmax=vmax)
            
            if save_plots or show_plots:
                plt.imshow(depth_map_golden, vmin=vmin, vmax=vmax)
                plt.colorbar()
                if save_plots:
                    plt.savefig(os.path.join(iter_folder,'finite_differences_'+str(i)+'.png'))
                if show_plots:
                    plt.show()
                else:
                    plt.close()
                
            loss = criterion(forward_model.forward_torch(torch.from_numpy(depth_map_golden).to(aif.device), aif), defocus_stack_torch)
            losses.append(loss.item())
            print('Loss:',loss.item())
            print()
        else:
            if not adaptive_grid:
                a_b_init = None
                # TODO: change grid search to opt
                depth_map, Z, argmin_indices, all_losses = section_search.grid_search_opt(aif, defocus_stack_torch, min_Z=min_Z, max_Z=max_Z, num_Z=num_Z, beta=beta, proxy=dpt_proxy, gamma=gamma, similarity_penalty=similarity_penalty, last_dpt=last_dpt)
                
                if save_plots or show_plots:
                    plt.imshow(depth_map)
                    plt.colorbar()
                    if save_plots:
                        plt.savefig(os.path.join(iter_folder,'grid_search_'+str(i)+'.png'))
                    if show_plots:
                        plt.show()
                    else:
                        plt.close()
                    
                loss = criterion(forward_model.forward_torch(torch.from_numpy(depth_map).to(aif.device), aif), defocus_stack_torch)
                losses.append(loss.item())
                print('Loss:',loss.item())
                print()
            else:
                # set the a_b_init
                a = np.maximum(dpt.detach().cpu().numpy().copy() - grid_window, 0)
                b = dpt.detach().cpu().numpy().copy() + grid_window
                Z = None
                argmin_indices = None
                a_b_init = (a, b)
    
            
            
            depth_map_golden = section_search.golden_section_search(Z, argmin_indices, aif, defocus_stack_torch, window=gss_window, tolerance=gss_tol, a_b_init=a_b_init, beta=beta, proxy=dpt_proxy, gamma=gamma, similarity_penalty=similarity_penalty, last_dpt=last_dpt)
    
            if save_plots or show_plots:
                plt.imshow(depth_map_golden, vmin=vmin, vmax=vmax)
                plt.colorbar()
                if save_plots:
                    plt.savefig(os.path.join(iter_folder,'golden_section_search_'+str(i)+'.png'))
                if show_plots:
                    plt.show()
                else:
                    plt.close()

        # TV thresh multiplier
        if tv_thresh_multiplier is not None:
            tv_thresh *= tv_thresh_multiplier
            if tv_thresh <= tv_thresh_min:
                tv_thresh = tv_thresh_min
        
        # # OPTIONAL -- remove outliers
        # if remove_outliers:
        #     print('tv_thresh =',tv_thresh)
        #     depth_map_golden, _ = section_search.remove_outliers(depth_map_golden, aif.cpu(), diff_thresh = diff_thresh, tv_thresh = tv_thresh, patch_type = outlier_patch_type)
        #     if save_plots or show_plots:
        #         plt.imshow(depth_map_golden, vmin=vmin, vmax=vmax)
        #         plt.colorbar()
        #         if save_plots:
        #             plt.savefig(os.path.join(iter_folder,'oulier_removal_'+str(i)+'.png'))
        #         if show_plots:
        #             plt.show()
        #         else:
        #             plt.close()
        
        dpt = torch.from_numpy(depth_map_golden).to(aif.device)
        
        
        loss = criterion(forward_model.forward_torch(dpt, aif), defocus_stack_torch)
        losses.append(loss.item())
        print('Loss:',loss.item(),', TV:',section_search.total_variation_torch(dpt).item())
        print('\nDPT result range: [',dpt.cpu().numpy().min(), ',', dpt.cpu().numpy().max(),']')

        
        # ---------------------
        # TV denoising (only for denoising experiment)
        # ------------------
        if not proxy_opt and dpt_denoising_weight != None and i > dpt_denoise_delay:
            dpt = torch.from_numpy(denoise_tv_bregman(dpt.cpu().numpy(), weight=dpt_denoising_weight)).to(aif.device)
            if save_plots:
                plt.imshow(dpt.cpu().numpy(), vmin=vmin, vmax=vmax)
                plt.colorbar()
                if save_plots:
                    plt.savefig(os.path.join(iter_folder,'dpt_denoise_'+str(i)+'.png'))
                if show_plots:
                    plt.show()
                else:
                    plt.close()
    
            loss = criterion(forward_model.forward_torch(dpt, aif), defocus_stack_torch)
            losses.append(loss.item())
            print('Loss after denoising:',loss.item(),', TV:',section_search.total_variation_torch(dpt).item())
            print()

        if proxy_opt:
            print('norm((dpt - proxy))^2:',(torch.norm(dpt - dpt_proxy.to(dpt.device))**2).item())
        print()
        print()
        return aif, dpt, dpt_proxy

    
    # -------------------------------------------------------------------------------------
    # Initialization
    dpt_proxy = None
    if depth_init is None:
        dpt = torch.full((width,height), 1).to(device, dtype=torch.float32)
        if proxy_opt:
            dpt_proxy = torch.full((width,height), 1).to(device, dtype=torch.float32)
    elif np.isscalar(depth_init):
        dpt = torch.full((width,height), depth_init).to(device, dtype=torch.float32)
        if proxy_opt:
            dpt_proxy = torch.full((width,height), depth_init).to(device, dtype=torch.float32)
    else:
        dpt = torch.from_numpy(depth_init).to(device, dtype=torch.float32)
        if proxy_opt:
            dpt_proxy = torch.from_numpy(depth_init).to(device, dtype=torch.float32)
    
    if not least_squares_first and aif_init is None:
        # aif as median filter of stack
        print('initializing aif to median filter of defocus stack')
        aif = np.median(defocus_stack, axis=0)
    else:
        aif = aif_init
        
            
    # rgb_weights = [0.2989, 0.5870, 0.1140]
    # dpt = np.dot(aif[..., :3], rgb_weights)
    # dpt = 0.7 + dpt * (1.9-0.7) / 255.
    # dpt = torch.from_numpy(dpt).to(device)
    # -------------------------------------------------------------------------------------
    
    if show_plots:
        if least_squares_first:
            plt.imshow(dpt.cpu().numpy())
            plt.title('DPT Initialization')
            plt.show()
        else:
            plt.imshow(aif/ IMAGE_RANGE)
            plt.title('AIF Initialization')
            plt.show()

    

    # last_dpt = None
    # if similarity_penalty and least_squares_first:
    last_dpt = torch.clone(dpt.detach())

    # coordinate descent
    last_loss = float('inf')
    for i in range(num_epochs):
        
        print('Iteration',i,'\n')
        
        if save_plots:
            iter_folder = os.path.join(experiment_folder,'iteration'+str(i))
            os.makedirs(iter_folder)
        else:
            iter_folder = None

        if i > 0:
            last_loss = criterion(forward_model.forward_torch(dpt, aif.to(dpt.device) if isinstance(aif, torch.Tensor) else torch.from_numpy(aif).to(dpt.device)), defocus_stack_torch).item()
        
        if least_squares_first:
            aif, dpt, dpt_proxy = generate_AIF(aif, dpt, dpt_proxy, ls_maxiter, iter_folder=iter_folder)
            aif, dpt, dpt_proxy = generate_DPT(aif, dpt, dpt_proxy, beta=beta, iter_folder=iter_folder, last_dpt=last_dpt)
        else:
            aif, dpt, dpt_proxy = generate_DPT(aif, dpt, dpt_proxy, beta=beta, iter_folder=iter_folder, last_dpt=last_dpt)
            aif, dpt, dpt_proxy = generate_AIF(aif, dpt, dpt_proxy, ls_maxiter, iter_folder=iter_folder)

        beta *= multiplier
        # if similarity_penalty:
        last_dpt = torch.clone(dpt.detach())

        if ls_maxiter_multiplier != None:
            ls_maxiter = int(ls_maxiter * ls_maxiter_multiplier)
            print('ls_maxiter updated to',ls_maxiter)
            # gss_tol = gss_tol / ls_maxiter_multiplier
            # print('gss_tol updated to', gss_tol)

        # Early stopping criterion
        loss = criterion(forward_model.forward_torch(dpt, aif.to(dpt.device) if isinstance(aif, torch.Tensor) else torch.from_numpy(aif).to(dpt.device)), defocus_stack_torch).item()
        # if loss > 1.05 * last_loss: # 5% greater
        #     print('Loss has started to diverge, terminated early at',i,'iters')
        #     break
        
        

        # loss plot
        if save_plots:
            plt.figure(figsize=(8, 4))
            x = np.arange(len(losses))
            dx = int(len(losses) / (i+1))
            plt.plot(x, losses)
            plt.scatter(x[1::dx], losses[1::dx], color='red', marker='x', s=100, label="clipped aif")
            plt.xticks(x[::dx], labels=np.arange(len(x))[::dx]) # 6 pts per iter
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Coordinate Descent")
            plt.savefig(os.path.join(experiment_folder,'loss.png'))
            plt.close()
    
            plt.figure(figsize=(8, 4))
            x = np.arange(len(losses))
            plt.plot(x, [math.log(loss, 10) for loss in losses])
            plt.scatter(x[1::dx], [math.log(loss, 10) for loss in losses[1::dx]], color='red', marker='x', s=100, label="clipped aif")
            plt.xticks(x[::dx], labels=np.arange(len(x))[::dx]) # 6 pts per iter
            plt.xlabel("Iteration")
            plt.ylabel("log(Loss)")
            plt.title("Coordinate Descent")
            plt.savefig(os.path.join(experiment_folder,'log_loss.png'))
            plt.close()

            with open(os.path.join(experiment_folder,"losses.txt"), "w") as file:
                for j in range(len(losses)):
                    if j % dx == 0:
                        file.write("iter "+str(j % dx)+":\n")
                    item = losses[j]
                    file.write(f"{item}\n")
        
        print()
        print()
        print('--------------------------')
        print()


    # OPTIONAL -- remove outliers
    if remove_outliers:
        dpt, _ = section_search.remove_outliers(dpt.cpu(), aif, diff_thresh = diff_thresh, tv_thresh = tv_thresh, patch_type = outlier_patch_type)
        if save_plots or show_plots:
            plt.imshow(dpt, vmin=vmin, vmax=vmax)
            plt.colorbar()
            if save_plots:
                plt.savefig(os.path.join(iter_folder,'oulier_removal_'+str(i)+'.png'))
            if show_plots:
                plt.show()
            else:
                plt.close()
        
    return dpt, aif, ls_maxiter

