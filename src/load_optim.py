"""
Load the desired optimizer.
"""

import torch.optim as optim
from sgd_lr_decay import SGDLRDecay

def load_optim(params, optim_method, eta0, alpha, milestones, nesterov,
               momentum, weight_decay):
    """
    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups.
        optim_method: which optimizer to use, currently only supports {'Adam',
            'SGD', 'SGD_Exp_Decay', 'SGD_1t_Decay', 'SGD_1sqrt_Decay',
            'SGD_Stage_Decay', 'SGD_ReduceLROnPlateau'}.
        eta0: starting step size.
        alpha: decaying factor for various methods.
        milestones: used for SGD stage decay denoting when to decrease the
            step size, unit in iteration.
        nesterov: whether to use nesterov momentum (True) or not (False).
        momentum: momentum factor used in variants of SGD.
        weight_decay: weight decay factor.

    Outputs:
        an optimizer
    """

    if optim_method == 'SGD' or optim_method == 'SGD_ReduceLROnPlateau':
        optimizer = optim.SGD(params=params, lr=eta0, momentum=momentum,
                              weight_decay=weight_decay, nesterov=nesterov)
    elif optim_method == 'Adam':
        optimizer = optim.Adam(params=params, lr=eta0,
                               weight_decay=weight_decay)
    elif optim_method.startswith('SGD') and optim_method.endswith('Decay'):
        if optim_method == 'SGD_Exp_Decay':
            scheme = 'exp'
        elif optim_method == 'SGD_1t_Decay':
            scheme = '1t'
        elif optim_method == 'SGD_1sqrt_Decay':
            scheme = '1sqrt'
        elif optim_method == 'SGD_Stage_Decay':
            scheme = 'stage'
        optimizer = SGDLRDecay(params=params, scheme=scheme, eta0=eta0,
                               alpha=alpha, milestones=milestones,
                               momentum=momentum, weight_decay=weight_decay,
                               nesterov=nesterov)
    else:
        raise ValueError("Invalid optimizer: {}".format(optim_method))

    return optimizer
