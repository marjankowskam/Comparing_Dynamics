import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS, SGD

# My hand-written optimziers
# try:
#     from .opts.gd import GD
#     from .opts.adam import Adam_
#     from .opts.adam_lbfgs import Adam_LBFGS
#     from .opts.adam_lbfgs_nncg import Adam_LBFGS_NNCG
#     from .opts.adam_lbfgs_gd import Adam_LBFGS_GD
#     from .opts.sam import SAM
#     from .opts.adam_sam import Adam_SAM
# except ModuleNotFoundError:
#     GD = Adam_ = Adam_LBFGS = Adam_LBFGS_NNCG = None
#     Adam_LBFGS_GD = SAM = Adam_SAM = None
import random
import re
import wandb

"""
Helper function for initializing the optimizer with specified parameters. 

INPUT: 
- opt_name: string; name of the optimizer
- opt_params: dictionary; arguments used to initialize the optimizer
- model_params: dictionary; contains Tensors of the model to be optimized
OUTPUT: 
- opt: torch.optim.Optimizer instance
"""
def get_opt(opt_name, opt_params_dict, model_params, n_examples=None):
    opt_params_list = [item for kv in opt_params_dict.items() for item in (kv[0], str(kv[1]))] #converting dictionary into a flat list of strings
    opt_params = get_opt_params(opt_params_list)

    if opt_name == 'adam':
        return Adam(model_params, **opt_params_dict)
    elif opt_name == "gd":
        # if GD is None:
        #     raise RuntimeError("The custom GD optimizer is not available in this checkout.")
        # make sure full batch data is passed here!!!
        return SGD(model_params, **opt_params)
    elif opt_name == 'sgd':
        return SGD(model_params, **opt_params)
    elif opt_name == 'lbfgs':
        if "history_size" in opt_params:
            opt_params["history_size"] = int(opt_params["history_size"])
        return LBFGS(model_params, **opt_params, line_search_fn='strong_wolfe')
    elif opt_name == 'sam':
        if SAM is None:
            raise RuntimeError("The custom SAM optimizer is not available in this checkout.")
        opt_params = opt_params_dict.copy()

        OPTIMIZER_MAP = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
        }

        base_optimizer_name = opt_params.pop("base_optimizer", "sgd")
        base_optimizer = OPTIMIZER_MAP[base_optimizer_name.lower()]

        return SAM(model_params, base_optimizer, **opt_params)
        
    elif opt_name == 'adam_sam':
        if Adam_SAM is None:
            raise RuntimeError("The custom Adam_SAM optimizer is not available in this checkout.")
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
    elif opt_name == "langevin":
        return Langevin(model_params, **opt_params_dict)
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


""" self-mplemented optimizer for Langevin dynamics """
class Langevin(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, temperature=1.0, n_examples=1):
        defaults = dict(
            lr=lr,
            temperature=temperature,
            n_examples=n_examples,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            T = group["temperature"]
            N = group["n_examples"]

            noise_scale = (2.0 * lr * T / N) ** 0.5

            for p in group["params"]:
                if p.grad is None:
                    continue

                p.add_(p.grad, alpha=-lr)

                if T > 0:
                    p.add_(torch.randn_like(p) * noise_scale)

        return loss