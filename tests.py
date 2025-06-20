
import forward_model
import utils

import least_squares

import torch

def test_forward_model():
    
    aif, dpt, gt_defocus_stack = utils.load_single_sample()

    # test forward model (NO torch)
    defocus_stack = forward_model.forward(dpt, aif)
    utils.plot_stacks_side_by_side(gt_defocus_stack, defocus_stack, globals.Df)

    # test forward_model (torch)
    defocus_stack = forward_model.forward_torch(dpt, aif)
    utils.plot_stacks_side_by_side(gt_defocus_stack, defocus_stack, globals.Df)



def test_least_squares():
    criterion = torch.nn.MSELoss()

    aif, dpt, gt_defocus_stack = utils.load_single_sample(res='half')
    
    # least squares
    defocus_stack = forward_model.forward(dpt, aif)

    recon_aif = least_squares.least_squares(dpt, defocus_stack)
    utils.plot_compare_rgb(recon_aif, aif)
    plt.show()

    print('Norm between recon and gt aif:', np.linalg.norm(np.array(recon_aif) - np.array(aif)))

    defocus_stack_pred = forward_model.forward_torch(dpt, torch.from_numpy(recon_aif))
    loss = criterion(defocus_stack_pred, torch.from_numpy(defocus_stack))

    print('MSE loss:',loss)


def test_least_squares_synthetic():    

    criterion = torch.nn.MSELoss()

    aif, dpt, gt_defocus_stack = utils.load_single_sample(res='half')
    
    # least squares
    defocus_stack = forward_model.forward(dpt, aif)

    print('Norm between my/their defocus stack:', np.linalg.norm(np.array(defocus_stack) - np.array(gt_defocus_stack)))
    print('MSE between stacks', np.mean((np.array(defocus_stack) - np.array(gt_defocus_stack))**2))


    recon_aif = least_squares.least_squares(dpt, gt_defocus_stack)
    utils.plot_compare_rgb(recon_aif, aif)
    plt.show()

    print('Norm between recon and gt aif:', np.linalg.norm(np.array(recon_aif) - np.array(aif)))

    defocus_stack_pred = forward_model.forward_torch(dpt, torch.from_numpy(recon_aif))
    loss = criterion(defocus_stack_pred, torch.from_numpy(defocus_stack))
    print('MSE loss:',loss)
