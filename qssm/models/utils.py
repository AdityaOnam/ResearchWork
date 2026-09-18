'''
Created on 1 May 2018

Utility functions.

@author: Miguel Molina Romero, Technical University of Munich
@contact: miguel.molina@tum.de
@License: LPGL
'''

import numpy as np
import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

def h_swish(x):
    return x * torch.nn.functional.relu6(x + 3, inplace=True) / 6

def h_sigmoid(x):
    return torch.nn.functional.relu6(x + 3, inplace=True) / 6

class Hswish(nn.Module):
    def forward(self, x):
        return h_swish(x)

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    return new_v



class DryException(Exception):
    def __init__(self, message, errors=0):

        # Call the base class constructor with the parameters it needs
        super(DryException, self).__init__(message)

        # Now for your custom code...
        self.errors = errors


def generate_synthetic_data(bfile):
    '''It generates free-water contaminated synthetic data for training.

    param bfile: b-values file.

    rtype: numpy matrices
    return: X, containing the contaminated signal, and Y, containing the
            free-water volume fraction
    '''

    N = 50000

    if bfile is None:
        raise DryException('generate_synthetic_data did \
                            not raise an excpetion')

    bvals = np.loadtxt(bfile, float, delimiter=' ')
    numbs = bvals.size

    Dfw = 3e-3
    Sfw = np.exp(-bvals * Dfw)
    Sfw = np.tile(Sfw, (N, 1))
    
    ffw = np.random.uniform(size=(N, 1))
    ft = 1 - ffw

    St = np.random.uniform(size=(N, numbs))
    St[:, np.where(bvals == 0)] = 1

    S = np.multiply(St, ft) + np.multiply(Sfw, ffw)

    return {'S': S, 'f': ffw}
