import torch
import torch.nn as nn
from torch.func import functional_call
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from tqdm import tqdm
import sys

from config import Config

from src.models import get_model
from src.train_utils import get_task, get_opt_params, get_opt, init_weights, set_random_seed
from hessian import get_hessian_experiment
from src.loss_balancers import get_loss_balancer


from src.train_utils import *

def normalization(v):
    return v / (v.norm() + 1e-12)


def train_and_collect_metrics(cfg, device, use_wandb):

  # initializations
  set_random_seed(cfg.initial_seed)

  # get model
  model = get_model(cfg.model_name, cfg.model_params).to(device)
  model.apply(init_weights)

  #get task
  print(cfg.task_name)
  print(cfg.task_params)
  task = get_task(cfg.task_name, cfg.task_params, device)

  # prepare named params template (keeps order)
  named_param_items = tuple(model.named_parameters())  # tuple of (name, param)
  param_names = [n for n, _ in named_param_items]
  param_template = [p.detach().clone() for _, p in named_param_items]  # templates for shapes
  old_params = torch.cat([p.detach().clone().view(-1) for _, p in named_param_items])

  # Task initialization
  task.initialize(param_names, model)
  full_data = task.get_full_data()

  # Optimizer
  opt = get_opt(cfg.opt_name, cfg.opt_params, model.parameters())

  # test/eval data
  data_test, test_targets = task.get_test_data()

  # Metrics to record - Loss and Grad norm
  loss_hist = {name: [] for (name, value) in task.initial_loss_weights().items()}
  grad_norm_hist = []
  update_norm = []

  def hessian_loss_fn(data, y, reduction='mean'):
    return task.balanced_loss_fn(data, y, alphas, reduction=reduction)

  # Eigenvector at the beginning
  model.eval()

  hessian = get_experiment_measurments(exp_name=cfg.experiment_name, model=model, predict=task.predict,
                              loss_fn = hessian_loss_fn,
                              device=device, named_param_items=named_param_items, 
                              data=full_data, task=task, optimizer = opt,
                              use_wandb=use_wandb, exp_params=cfg.experiment_params)
  params_for_hessian_log = torch.cat([p.detach().clone().view(-1) for _, p in named_param_items])
  
# TRAING LOOP
  for i in (progress_bar:=tqdm(range(cfg.num_epochs), ncols=125)):

    model.train()
    data = task.get_data()

    # Optimizer step
    def closure():
        nonlocal grad_vec
        opt.zero_grad()
        outputs = task.predict(data, model)
        loss = task.balanced_loss_fn(data, outputs, alphas)
        loss.backward()
        return loss

    if isinstance(opt, SAM):
        # SAM requires two forward-backward passes
        loss = closure()          # compute gradients
        opt.first_step()          # ascent step
        closure()                 # compute gradients at perturbed weights
        opt.second_step()         # descent step
    else:
        opt.step(closure)

    # evaluation
    model.eval()

    grad_norm = grad_vec.norm().item()
    grad_vec = grad_vec / (grad_norm + 1e-12)
    grad_norm_hist.append(grad_norm)

    model.zero_grad(set_to_none=True)

    loss = task.balanced_loss_fn(data, task.predict(data, model), alphas)
    if use_wandb:
        wandb.log({
            "loss": loss.item(),
            "loss_log10": np.log10(loss.item()+1e-12),
            "grad_norm": grad_norm,
            "grad_norm_log10": np.log10(grad_norm + 1e-12),
        }, step = i)

    params_for_hessian_log = torch.cat([p.detach().clone().view(-1) for _, p in named_param_items]).to(device) #CAN BE MOVED BELOW THE DATA COLLECTION
    with torch.no_grad():
      param_diff = params_for_hessian_log - old_params
      param_update_norm = torch.norm(param_diff).item()
      old_params = params_for_hessian_log.detach()

    hessian.collect_metrics(model_params=params_for_hessian_log, step_id=i, batch_data=data, grad_vec=grad_vec, param_diff=param_diff)

    update_norm.append(param_update_norm)
    # progress_bar.set_postfix_str("res {:.2e} / bc {:.2e} / ic {:.2e} / grad_norm {:.2e}".format(loss_res_hist[-1], loss_bc_hist[-1], loss_ic_hist[-1], grad_norm))
    postfix = " / ".join(f"{name} {hist[-1]:.2e}" for name, hist in loss_hist.items())
    postfix += f" / grad_norm {grad_norm:.2e}"
    progress_bar.set_postfix_str(postfix)


    # eval metrics
  metrics = task.evaluate(model, device)
#   print(f"final loss {loss_hist[-1]} / test performance {metrics}")

  run_details = {'model_state_dict': model.state_dict(),
    'optimizer_state_dict': opt.state_dict(),
    'loss_hist': loss_hist,
    'grad_norm_hist': grad_norm_hist,
    'test_metrics': metrics,
    'hessian_object': hessian,
    'update_norm': update_norm
    }

  return run_details
