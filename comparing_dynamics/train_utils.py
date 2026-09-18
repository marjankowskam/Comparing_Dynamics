import numpy as np
import torch
import torch.nn as nn
import random
import re
import wandb

from .tasks.mnist import MNISTTask
from .tasks.fashionmnist import FashionMNISTTask

def get_task(task_name, task_params, device):
    return TASK_OPTIONS[task_name](**task_params, device=device)

TASK_OPTIONS = {
    "mnist": MNISTTask,
    "fashion_mnist": FashionMNISTTask,
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

# Source code: https://github.com/mkerrwinter/glass_net_publication/blob/main/train_a_model.py
def train(dataloader, model, loss_fn, optimizer, device):
        
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y) 

        for param in model.parameters():
            param.grad = None  

        loss.backward() 
        optimizer.step() 
        
# Define training process with Gaussian noise
# Source code: https://github.com/mkerrwinter/glass_net_publication/blob/main/train_a_model.py
def train_Langevin(dataloader, model, loss_fn, optimizer, device, kT, lr, 
                   N_examples):
        
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y) 

        for param in model.parameters():
            param.grad = None  

        loss.backward() 
        optimizer.step() 

        # Add noise given by temperature T
        strength = np.sqrt(2.0*kT*lr/N_examples)
        with torch.no_grad():
            for param in model.parameters():
                noise_tensor = torch.randn(param.size()) * strength
                noise_tensor = noise_tensor.to(device)
                param.add_(noise_tensor)