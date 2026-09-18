from inspect import signature
from collections import namedtuple, OrderedDict
from typing import Type, Any, Callable, Union, List, Optional

import math
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from utils import SeparableConv2d, h_swish, _make_divisible, h_sigmoid, Hswish

from building_blocks import *
#from timm.models.efficientnet_blocks import InvertedResidual, DepthwiseSeparableConv
from timm.models.layers import create_conv2d, drop_path, make_divisible, create_act_layer


def get_init_block(planes, block_type='default', args=None):
    assert args is not None
    separable = args.get('separable', False)

    if block_type == 'BasicBlock':
        return BasicBlock(planes, planes, separable=separable)
    elif block_type == 'Bottleneck':
        return Bottleneck(planes, planes, separable=separable, expansion=1)
    elif block_type == 'BasicDense':
        return BasicDenseLayer(planes, bn_size=2)
    elif block_type == 'DwConv':
        k = args.get('dw_kernel_size', 3)
        return nn.Conv2d(planes, planes, kernel_size=k, stride=1, padding=1, groups=planes)
    elif block_type == 'FullConv':
        k = args.get('dw_kernel_size', 3)
        return nn.Conv2d(planes, planes, kernel_size=k, stride=1, padding=1)
    elif block_type == 'PwConv':
        return nn.Conv2d(planes, planes, kernel_size=1, stride=1)
    elif block_type == 'DartCell':
        return Cell(args['genotype'], args['C_prev_prev'], args['C_prev'], args['C_curr'],
                    args['reduction'], args['reduction_prev'])
    elif block_type == 'InvertedResidualCell':
        return InvertedResidual(args['in_chs'], args['out_chs'], args['dw_kernel_size'],
                                 stride=args['stride'], dilation=args['dilation'], pad_type=args['pad_type'],
                                 act_layer=args['act_layer'], noskip=args['noskip'], exp_ratio=args['exp_ratio'],
                                 exp_kernel_size=args['exp_kernel_size'], pw_kernel_size=args['pw_kernel_size'],
                                 se_layer=args.get('se_layer', None), norm_layer=args['norm_layer'],
                                 conv_kwargs=args.get('conv_kwargs', {}), drop_path_rate=args['drop_path_rate'])
    return nn.Identity()

class GlobalFeatureBlock_Diffusion(nn.Module):
    expansion: int = 1

    def __init__(self, planes, args):
        super(GlobalFeatureBlock_Diffusion, self).__init__()

        norm_layer = args.get('norm_layer', nn.BatchNorm2d)

        K = args.get('K', 10)
        nonlinear_pde = args.get('nonlinear_pde', True)
        assert nonlinear_pde is True

        self.cDx = nn.Parameter(torch.tensor(args.get('cDx', 1.0), dtype=torch.float32))
        self.cDy = nn.Parameter(torch.tensor(args.get('cDy', 1.0), dtype=torch.float32))
        self.dx = torch.tensor(args.get('dx', 1.0), dtype=torch.float32)
        self.dy = torch.tensor(args.get('dy', 1.0), dtype=torch.float32)
        # ✅ NOVELTY: The fixed dt is removed. We will learn a map.
        self.dt_min = args.get('dt_min', 0.01)

        self.use_f_for_g = args.get('use_f_for_g', False)
        use_silu = args.get('use_silu', False)
        self.use_res = args.get('use_res', False)
        use_cDs = args.get('use_cDs', False)
        use_dw = args.get('use_dw', False)
        self.use_dot = args.get('use_dot', False)
        self.no_f = args.get('no_f', False)
        self.constant_Dxy = args.get('constant_Dxy', False)
        block_type = args.get('cell_type', 'default')
        drop_path_rate = args.get('drop_path_rate', 0.)

        dw_kernel_size = args.get('dw_kernel_size', 3)
        pw_kernel_size = args.get('pw_kernel_size', 1)
        exp_kernel_size = args.get('exp_kernel_size', 1)
        se_layer = args.get('se_layer', None)
        old_style = args.get('old_style', False)

        dilation = args.get('dilation', 1)
        pad_type = args.get('pad_type', '')
        stride = args.get('stride', 1)
        in_chs = args.get('in_chs', planes)
        out_chs = args.get('out_chs', planes)
        if 'out_chs' in args:
            planes = out_chs

        self.pde_state = args.get('pde_state', 0)
        self.nonlinear_pde = nonlinear_pde
        self.K = K
        self.act = nn.SiLU(inplace=True) if use_silu else nn.ReLU(inplace=True)
        if 'act_layer' in args:
            self.act = args['act_layer'](inplace=True)
        
        self.init_h = get_init_block(planes, block_type, args)
        self.bn_out = norm_layer(planes)

        self.drop_path_rate = drop_path_rate
        self.stride = stride
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.planes = planes
        self.block_type = block_type
        
        if self.nonlinear_pde:
            self.convg = create_conv2d(planes, planes, kernel_size=dw_kernel_size, stride=1, dilation=dilation, padding=pad_type, depthwise=True)
            self.convg1 = create_conv2d(planes, planes, kernel_size=dw_kernel_size, stride=1, dilation=dilation, padding=pad_type, depthwise=True)
            self.bng = norm_layer(planes)
            self.bng1 = norm_layer(planes)
            if not self.constant_Dxy:
                self.convDx = create_conv2d(planes, planes, kernel_size=dw_kernel_size, stride=1, dilation=dilation, padding=pad_type, depthwise=True)
                self.convDy = create_conv2d(planes, planes, kernel_size=dw_kernel_size, stride=1, dilation=dilation, padding=pad_type, depthwise=True)
                self.bnDx = norm_layer(planes)
                self.bnDy = norm_layer(planes)

        # ✅ NOVELTY: Layer to generate a spatially adaptive time step map
        self.conv_dt = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=1),
            norm_layer(planes),
            self.act,
            nn.Conv2d(planes, 1, kernel_size=1),
            nn.Softplus() # Ensures dt is always positive
        )

    def feature_info(self, location):
        return dict(module='', hook_type='', num_chs=self.planes)

    def forward(self, s0, s1=None, drop_path=None):
        f = s1 if self.block_type == 'DartCell' else s0
        h = self.init_h(s0, s1, drop_path) if self.block_type == 'DartCell' else self.init_h(f)

        if (self.stride != 1) or (self.in_chs != self.out_chs):
            f = h
        residual = f
        
        h_prev = h

        g0 = f if self.use_f_for_g else h
        g = self.act(self.bng(self.convg(g0)))
        g1 = self.act(self.bng1(self.convg1(g0)))

        if self.constant_Dxy:
            Dx, Dy = torch.abs(self.cDx) + 1e-6, torch.abs(self.cDy) + 1e-6
        else:
            Dx = self.act(self.bnDx(self.convDx(h))) + 1e-6
            Dy = self.act(self.bnDy(self.convDy(h))) + 1e-6

        # ✅ NOVELTY: Generate spatially adaptive time step map
        dt_map = self.conv_dt(h) + self.dt_min

        # Calculate advection and diffusion terms
        advection_x = g * (torch.roll(h_prev, -1, dims=2) - torch.roll(h_prev, 1, dims=2)) / (2 * self.dx)
        advection_y = g1 * (torch.roll(h_prev, -1, dims=3) - torch.roll(h_prev, 1, dims=3)) / (2 * self.dy)
        
        diffusion_x = Dx * (torch.roll(h_prev, 1, dims=2) - 2 * h_prev + torch.roll(h_prev, -1, dims=2)) / (self.dx * self.dx)
        diffusion_y = Dy * (torch.roll(h_prev, 1, dims=3) - 2 * h_prev + torch.roll(h_prev, -1, dims=3)) / (self.dy * self.dy)

        for _ in range(self.K):
            update = (diffusion_x + diffusion_y) - (advection_x + advection_y)
            source = f if not self.no_f else 0.0
            h_prev = h_prev + dt_map * (update + source)
            
        h = h_prev
        h = self.bn_out(h)
        h = self.act(h)

        if self.use_res:
            if self.drop_path_rate > 0.:
                h = drop_path(h, self.drop_path_rate, self.training)
            h = h + residual

        return h


# ✅ Default configuration generator (kept for completeness)
def get_default_pde_args(in_chs=64, out_chs=64, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU):
    return {
        'K': 5,
        'nonlinear_pde': True,
        'separable': True,
        'pde_state': 0,
        'dx': 1,
        'dy': 1,
        'dt': 0.2,
        'use_f_for_g': False,
        'use_diff_eps': True,
        'use_silu': False,
        'use_res': True,
        'use_cDs': False,
        'use_dw': False,
        'use_dot': False,
        'drop_path_rate': 0.,
        'constant_Dxy': True,
        'no_f': False,
        'cell_type': 'default',
        'custom_uv': '',
        'custom_dxy': '',
        'norm_layer': norm_layer,
        'act_layer': act_layer,
        'pad_type': 'same',
        'in_chs': in_chs,
        'out_chs': out_chs,
        'dw_kernel_size': 3,
        'pw_kernel_size': 1,
        'exp_kernel_size': 1,
    }
