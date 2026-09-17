import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS, SGD

from .opts.gd import GD
from .opts.adam import Adam_
from .opts.adam_lbfgs import Adam_LBFGS
from .opts.adam_lbfgs_nncg import Adam_LBFGS_NNCG
from .opts.adam_lbfgs_gd import Adam_LBFGS_GD
from .opts.sam import SAM
from .opts.adam_sam import Adam_SAM
import random
import re
import wandb

LOG_FREQ = 20 # Hard-coded for now -- this is done to match the max_iter of the LBFGS optimizer + save time

from tasks.mnist import MNISTTask
from tasks.pde_regression import PDERegressionTask
from tasks.pinn import PINNTask
from tasks.fashionmnist import FashionMNISTTask


def get_task(task_name, task_params, device):
    return TASK_OPTIONS[task_name](**task_params, device=device)

TASK_OPTIONS = {
    "mnist": MNISTTask,
    "fashion_mnist": FashionMNISTTask,
    "pde_regression": PDERegressionTask,
    "pinn": PINNTask
}


"""
Helper function for setting seed for the random number generator in various packages.

INPUT: 
- seed: integer
"""
def set_random_seed(seed): 
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

"""
Helper function for initializing neural net parameters. 
"""
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

# """
# Helper function for making predictions with PINN. 

# INPUT: 
# - x: tuple of (x_res, x_left, x_upper, x_lower)
# - t: tuple of (t_res, t_left, t_upper, t_lower)
# - model: PINN model
# OUTPUT: 
# - preds: tuple of (pred_res, pred_left, pred_upper, pred_lower)
# where: 
# > pred_res: predictions on residual points
# > pred_left: predictions on initial points
# > pred_upper: predictions on upper boundary points
# > pred_lower: predictions on lower boundary points
# """
# def predict(x, t, model): 
#     x_res, x_left, x_upper, x_lower = x
#     t_res, t_left, t_upper, t_lower = t
    
#     pred_res = model(x_res, t_res)
#     pred_left = model(x_left, t_left)
#     pred_upper = model(x_upper, t_upper)
#     pred_lower = model(x_lower, t_lower)
    
#     preds = (pred_res, pred_left, pred_upper, pred_lower) 
    
#     return preds


"""
Helper function for initializing the optimizer with specified parameters. 

INPUT: 
- opt_name: string; name of the optimizer
- opt_params: dictionary; arguments used to initialize the optimizer
- model_params: dictionary; contains Tensors of the model to be optimized
OUTPUT: 
- opt: torch.optim.Optimizer instance
"""
def get_opt(opt_name, opt_params_dict, model_params):
    opt_params_list = [item for kv in opt_params_dict.items() for item in (kv[0], str(kv[1]))] #converting dictionary into a flat list of strings
    if opt_name == "lbfgs":
        opt_params_list = ["history_size", str(cfg.opt_params['lbfgs_history_size'])]
    opt_params = get_opt_params(opt_params_list)

    if opt_name == 'adam':
        return Adam_(model_params, **opt_params_dict)
    elif opt_name == "gd":
        return GD(model_params, **opt_params)
    elif opt_name == 'sgd':
        return SGD(model_params, **opt_params)
    elif opt_name == 'lbfgs':
        if "history_size" in opt_params:
            opt_params["history_size"] = int(opt_params["history_size"])
        return LBFGS(model_params, **opt_params, line_search_fn='strong_wolfe')
    elif opt_name == 'adam_lbfgs':
        if "switch_epochs" not in opt_params:
            raise KeyError("switch_epochs is not specified for Adam_LBFGS optimizer.")
        switch_epochs = opt_params["switch_epochs"]

        # Ensure switch_epochs is a list of integers
        if not isinstance(switch_epochs, list):
            switch_epochs = [switch_epochs]
        switch_epochs = [int(epoch) for epoch in switch_epochs]

        # Get parameters for Adam and LBFGS, remove the prefix "adam_" and "lbfgs_" from the keys
        adam_params = {k[5:]: v for k, v in opt_params.items() if k.startswith("adam_")}
        lbfgs_params = {k[6:]: v for k, v in opt_params.items() if k.startswith("lbfgs_")}
        lbfgs_params["line_search_fn"] = "strong_wolfe"
        
        # If max_iter or history_size is specified, convert them to integers
        if "max_iter" in lbfgs_params:
            lbfgs_params["max_iter"] = int(lbfgs_params["max_iter"])
        if "history_size" in lbfgs_params:
            lbfgs_params["history_size"] = int(lbfgs_params["history_size"])

        return Adam_LBFGS(model_params, switch_epochs, adam_params, lbfgs_params)
    elif opt_name == 'adam_lbfgs_nncg':
        if "switch_epoch_lbfgs" not in opt_params:
            raise KeyError("switch_epoch_lbfgs is not specified for Adam_LBFGS_NNCG optimizer.")
        if "switch_epoch_nncg" not in opt_params:
            raise KeyError("switch_epoch_nncg is not specified for Adam_LBFGS_NNCG optimizer.")
        if "precond_update_freq" not in opt_params:
            raise KeyError("precond_update_freq is not specified for Adam_LBFGS_NNCG optimizer.")
        switch_epoch_lbfgs = int(opt_params["switch_epoch_lbfgs"])
        switch_epoch_nncg = int(opt_params["switch_epoch_nncg"])
        precond_update_freq = int(opt_params["precond_update_freq"])

        # Get parameters for Adam, LBFGS, and NNCG, remove the prefix "adam_", "lbfgs_", and "nncg_" from the keys
        adam_params = {k[5:]: v for k, v in opt_params.items() if k.startswith("adam_")}
        lbfgs_params = {k[6:]: v for k, v in opt_params.items() if k.startswith("lbfgs_")}
        nncg_params = {k[5:]: v for k, v in opt_params.items() if k.startswith("nncg_")}
        lbfgs_params["line_search_fn"] = "strong_wolfe"
        nncg_params["line_search_fn"] = "armijo"

        nncg_params["verbose"] = True

        # If max_iter or history_size is specified, convert them to integers
        if "max_iter" in lbfgs_params:
            lbfgs_params["max_iter"] = int(lbfgs_params["max_iter"])
        if "history_size" in lbfgs_params:
            lbfgs_params["history_size"] = int(lbfgs_params["history_size"])
        if "rank" in nncg_params:
            nncg_params["rank"] = int(nncg_params["rank"])

        return Adam_LBFGS_NNCG(model_params, switch_epoch_lbfgs, switch_epoch_nncg, precond_update_freq, adam_params, lbfgs_params, nncg_params)
    elif opt_name == 'adam_lbfgs_gd':
        if "switch_epoch_lbfgs" not in opt_params:
            raise KeyError("switch_epoch_lbfgs is not specified for Adam_LBFGS_GD optimizer.")
        if "switch_epoch_gd" not in opt_params:
            raise KeyError("switch_epoch_gd is not specified for Adam_LBFGS_GD optimizer.")
        switch_epoch_lbfgs = int(opt_params["switch_epoch_lbfgs"])
        switch_epoch_gd = int(opt_params["switch_epoch_gd"])

        # Get parameters for Adam, LBFGS, and GD, remove the prefix "adam_", "lbfgs_", and "gd_" from the keys
        adam_params = {k[5:]: v for k, v in opt_params.items() if k.startswith("adam_")}
        lbfgs_params = {k[6:]: v for k, v in opt_params.items() if k.startswith("lbfgs_")}
        gd_params = {k[3:]: v for k, v in opt_params.items() if k.startswith("gd_")}
        lbfgs_params["line_search_fn"] = "strong_wolfe"
        gd_params["line_search_fn"] = "armijo"

        # If max_iter or history_size is specified, convert them to integers
        if "max_iter" in lbfgs_params:
            lbfgs_params["max_iter"] = int(lbfgs_params["max_iter"])
        if "history_size" in lbfgs_params:
            lbfgs_params["history_size"] = int(lbfgs_params["history_size"])

        return Adam_LBFGS_GD(model_params, switch_epoch_lbfgs, switch_epoch_gd, adam_params, lbfgs_params, gd_params)
    elif opt_name == 'sam':
        opt_params = opt_params_dict.copy()

        OPTIMIZER_MAP = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
        }

        base_optimizer_name = opt_params.pop("base_optimizer", "sgd")
        base_optimizer = OPTIMIZER_MAP[base_optimizer_name.lower()]

        return SAM(model_params, base_optimizer, **opt_params)
        
    elif opt_name == 'adam_sam':
        if "switch_epochs" not in opt_params:
            raise KeyError("switch_epochs is not specified for Adam_LBFGS optimizer.")
        switch_epochs = opt_params["switch_epochs"]

        # Ensure switch_epochs is a list of integers
        if not isinstance(switch_epochs, list):
            switch_epochs = [switch_epochs]
        switch_epochs = [int(epoch) for epoch in switch_epochs]

        # Get parameters for Adam and LBFGS, remove the prefix "adam_" and "lbfgs_" from the keys
        adam_params = {k[5:]: v for k, v in opt_params.items() if k.startswith("adam_")}

        return Adam_SAM(model_params, switch_epochs, adam_params)
    else:
        raise ValueError(f'Optimizer {opt_name} not supported')

"""
Helper function for parsing a mixed list of strings and numerical values. 

INPUT: 
- params_list: list of strings
OUTPUT: 
- params_dict: dictionary
"""
def parse_params_list(params_list): 
    # return an empty dictionary if there is no parameters specified
    if params_list is None: 
        return {}

    # parse parameter names and specified (if any) values
    params_dict = {}
    current_parameter = None
    match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
    for token in params_list: 
        # attempt to extract a number from the token
        parsed_number = re.search(match_number, token)
        # if no match is found, then the token is a parameter name
        if parsed_number is None:
            params_dict[token] = None
            current_parameter = token
        # if the token indeed is a number (integer, decimal, or in scientific notation)
        else: 
            # if the current parameter is not specified yet, then the number is the value of the current parameter
            # otherwise, the number is appended to the list of values associated with current parameter
            if params_dict[current_parameter] is not None:
                if not isinstance(params_dict[current_parameter], list):
                    params_dict[current_parameter] = [params_dict[current_parameter]]
                params_dict[current_parameter].append(float(parsed_number.group()))
            else:
                params_dict[current_parameter] = float(parsed_number.group())
    
    return params_dict

"""
Helper function for forming optimizer parameters. 

INPUT: 
- opt_params_list: list of strings
OUTPUT: 
- opt_params: dictionary
"""
def get_opt_params(opt_params_list): 
    return parse_params_list(opt_params_list)


"""
Helper function for getting the list of logging times.

INPUT:
- opt: optimizer to be used
- log_freq: integer; logging frequency
OUTPUT:
- log_times: list of positive integers; logging times
"""
# TODO: Make this robust to when log_freq does not match the max_iter of the LBFGS optimizer
def get_log_times(opt, log_freq, num_epochs):
    log_times = []
    if isinstance(opt, (Adam_LBFGS_NNCG, Adam_LBFGS_GD)):
        # Get times up to opt.switch_epoch1, starting at log_freq (Adam)
        log_times = list(range(log_freq - 1, opt.switch_epoch1, log_freq))
        # Get times up to opt.switch_epoch2 (possibly including opt.switch_epoch2), starting at opt.switch_epoch1 (L-BFGS)
        log_times += list(range(opt.switch_epoch1, opt.switch_epoch2, 1))
        # Get times up to num_epochs (possibly including num_epochs), starting at opt.switch_epoch2 (NNCG/GD)
        log_times += list(range(opt.switch_epoch2 + log_freq - 1, num_epochs, log_freq))
    elif isinstance(opt, Adam_LBFGS):
        # Get times up to opt.switch_epochs[0], starting at log_freq (Adam)
        log_times = list(range(log_freq - 1, opt.switch_epochs[0], log_freq))
        # Get times up to num_epochs (possibly including num_epochs), starting at opt.switch_epochs[0] (L-BFGS)
        log_times += list(range(opt.switch_epochs[0], num_epochs, 1))
    elif isinstance(opt, Adam):
        # Get times up to num_epochs (possibly including num_epochs), starting at log_freq (Adam)
        log_times = list(range(log_freq - 1, num_epochs, log_freq))
    elif isinstance(opt, LBFGS):
        # Get all times up to num_epochs (possibly including num_epochs), starting at 1 (L-BFGS)
        log_times = list(range(0, num_epochs, 1))

    return log_times

# def train(model,
#           proj_name,
#           pde_name,
#           pde_params,
#           loss_name,
#           opt_name,
#           opt_params_list,
#           n_x,
#           n_t,
#           n_res,
#           num_epochs,
#           device):
#     model.apply(init_weights)

#     x_range, t_range, loss_func, pde_coefs = get_pde(pde_name, pde_params, loss_name)
#     opt_params = get_opt_params(opt_params_list)
#     opt = get_opt(opt_name, opt_params, model.parameters())

#     logging_times = get_log_times(opt, LOG_FREQ, num_epochs)

#     x, t, data_params = get_data(x_range, t_range, n_x, n_t, random=True, num_res_samples=n_res, device=device)
#     wandb.log({'x': x, 't': t}) # Log training set

#     loss_res, loss_bc, loss_ic = loss_func(x, t, predict(x, t, model))
#     loss = loss_res + loss_bc + loss_ic
#     wandb.log({'loss': loss.item(),
#             'loss_res': loss_res.item(),
#             'loss_bc': loss_bc.item(),
#             'loss_ic': loss_ic.item()})
    
#     for i in range(num_epochs):
#         model.train()

#         # Update the preconditioner for NysNewtonCG
#         if isinstance(opt, Adam_LBFGS_NNCG) and i >= opt.switch_epoch2 and i % opt.precond_update_freq == 0:
#             opt.zero_grad()
#             outputs = predict(x, t, model)
#             loss_res, loss_bc, loss_ic = loss_func(x, t, outputs)
#             loss = loss_res + loss_bc + loss_ic
#             grad_tuple = torch.autograd.grad(
#                 loss, model.parameters(), create_graph=True)
#             opt.nncg.update_preconditioner(grad_tuple)

#         # Separate closure is needed for NysNewtonCG/GD
#         if isinstance(opt, (Adam_LBFGS_NNCG, Adam_LBFGS_GD)) and i >= opt.switch_epoch2:
#             def closure():
#                 opt.zero_grad()
#                 outputs = predict(x, t, model)
#                 loss_res, loss_bc, loss_ic = loss_func(x, t, outputs)
#                 loss = loss_res + loss_bc + loss_ic
#                 grad_tuple = torch.autograd.grad(loss, model.parameters(), create_graph=True)
#                 return loss, grad_tuple
#         else:
#             def closure():
#                 opt.zero_grad()
#                 outputs = predict(x, t, model)
#                 loss_res, loss_bc, loss_ic = loss_func(x, t, outputs)
#                 loss = loss_res + loss_bc + loss_ic
#                 loss.backward()
#                 return loss

#         if isinstance(opt, (Adam_LBFGS_NNCG, Adam_LBFGS_GD)) and i >= opt.switch_epoch2:
#             grad = opt.step(closure)
#         # elif isinstance(opt, SAM):
#         #     loss = closure()
#         #     opt.first_step()
#         #     closure()
#         #     opt.second_step()
#         else:
#             opt.step(closure)

#         # record model parameters and loss
#         model.eval()
#         if i in logging_times:
#             loss_res, loss_bc, loss_ic = loss_func(x, t, predict(x, t, model))
#             loss = loss_res + loss_bc + loss_ic

#             # Compute the gradient norm of the full objective function
#             # NOTE: This will not work if we do minibatching
#             if isinstance(opt, (Adam_LBFGS_NNCG, Adam_LBFGS_GD)) and i >= opt.switch_epoch2:
#                 grad_norm = torch.norm(grad).item()
#             else:
#                 grad_norm = 0
#                 for p in model.parameters():
#                     grad_norm += p.grad.norm().item() ** 2
#                 grad_norm = grad_norm ** 0.5

#             if isinstance(opt, Adam_LBFGS_NNCG) and i >= opt.switch_epoch2:
#                 wandb.log({'step_size': opt.nncg.state_dict()['state'][0]['t']},
#                             commit=False)
#             elif isinstance(opt, Adam_LBFGS_GD) and i >= opt.switch_epoch2:
#                 wandb.log({'step_size': opt.gd.state_dict()['state'][0]['t']},
#                             commit=False)

#             wandb.log({'loss': loss.item(),
#                 'loss_res': loss_res.item(),
#                 'loss_bc': loss_bc.item(),
#                 'loss_ic': loss_ic.item(),
#                 'grad_norm': grad_norm})
    
#     # evaluate errors
#     with torch.no_grad():
#         predictions = torch.vstack(predict(x, t, model)).cpu().detach().numpy()
#     targets = get_ref_solutions(pde_name, pde_coefs, x, t, data_params)
#     train_l1re = l1_relative_error(predictions, targets)
#     train_l2re = l2_relative_error(predictions, targets)

#     # coarse grid for testing
#     n_x_test = int((n_x - 1) / 2) + 1
#     n_t_test = n_t
#     x_test, t_test, data_params_test = get_data(x_range, t_range, n_x_test, n_t_test, random=False, device=device)
#     with torch.no_grad():
#         predictions = torch.vstack(predict(x_test, t_test, model)).cpu().detach().numpy()
#     targets = get_ref_solutions(pde_name, pde_coefs, x_test, t_test, data_params_test)
#     test_l1re = l1_relative_error(predictions, targets)
#     test_l2re = l2_relative_error(predictions, targets)

#     wandb.log({'train/l1re': train_l1re, 
#                 'train/l2re': train_l2re, 
#                 'test/l1re': test_l1re, 
#                 'test/l2re': test_l2re})
