from typing import List
import torch
from torch import nn

def add_weight_decay(model: nn.Module, 
                     weight_decay: float, 
                     skip_list: List=()) -> List:
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  #frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def weight_decay_val(paramlist):
    """
    computes the value of the weight decay term after an epoch
    """
    sum = 0.
    for weight in paramlist[1]['params']:
        with torch.no_grad():
            sum += (weight**2).sum()
    return sum.item()

def weight_decay_eval(paramlist):
    """
    computes the value of the weight decay term after an epoch
    """
    sum = 0.
    for weight in paramlist[1]['params']:
        with torch.no_grad():
            sum += (weight**2).sum()
    return sum

def Llayers(L,d,width):
    #construct L-1 linear layers; bias term only on last linear layer
    if L < 2:
        raise ValueError("L must be at least 2")
    if L == 2:
        linear_layers = [nn.Linear(d,width,bias=True)]
    if L > 2:
        linear_layers = [nn.Linear(d,width,bias=False)]
        for l in range(L-3):
            linear_layers.append(nn.Linear(width,width,bias=False))
        linear_layers.append(nn.Linear(width,width,bias=True))

    relu = nn.ReLU()

    last_layer = nn.Linear(width,1)

    layers = linear_layers + [relu,last_layer]

    return nn.Sequential(*layers)
