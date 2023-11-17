'''
Function:
    Build activation functions

'''
import copy
import torch.nn as nn
'''BuildActivation'''


def BuildActivation(act_cfg):
    supported_activations = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'relu6': nn.ReLU6,
        'prelu': nn.PReLU,
        'sigmoid': nn.Sigmoid,
        'identity': nn.Identity,
        'leakyrelu': nn.LeakyReLU,
    }
    selected_act_func = supported_activations[act_cfg['type']]
    act_cfg = copy.deepcopy(act_cfg)
    act_cfg.pop('type')
    return selected_act_func(**act_cfg)
