from inspect import signature
from collections import namedtuple, OrderedDict
from typing import Type, Any, Callable, Union, List, Optional

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from utils import SeparableConv2d, h_swish, _make_divisible, h_sigmoid, Hswish
from building_blocks import *
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
        return InvertedResidual(
            args['in_chs'], args['out_chs'], args['dw_kernel_size'],
            stride=args['stride'], dilation=args['dilation'], pad_type=args['pad_type'],
            act_layer=args['act_layer'], noskip=args['noskip'], exp_ratio=args['exp_ratio'],
            exp_kernel_size=args['exp_kernel_size'], pw_kernel_size=args['pw_kernel_size'],
            se_layer=args.get('se_layer', None), norm_layer=args['norm_layer'],
            conv_kwargs=args.get('conv_kwargs', {}), drop_path_rate=args['drop_path_rate']
        )
    return nn.Identity()


class GlobalFeatureBlock_Diffusion(nn.Module):
    expansion: int = 1

    def __init__(self, planes, args):
        super().__init__()

        norm_layer = args.get('norm_layer', nn.BatchNorm2d)

        self.K = args.get('K', 10)
        self.dx = torch.tensor(args.get('dx', 1.0), dtype=torch.float32)
        self.dy = torch.tensor(args.get('dy', 1.0), dtype=torch.float32)
        self.dt_min = args.get('dt_min', 0.01)
        self.dt_max = args.get('dt_max', 0.1)
        # dt_live=False reproduces the published behaviour EXACTLY, including the dead
        # timestep -- see the conv_dt comment below. Only set it True deliberately.
        self.dt_live = args.get('dt_live', False)

        self.use_res = args.get('use_res', False)
        self.no_f = args.get('no_f', False)
        self.constant_Dxy = args.get('constant_Dxy', False)

        block_type = args.get('cell_type', 'default')
        drop_path_rate = args.get('drop_path_rate', 0.)

        dw_kernel_size = args.get('dw_kernel_size', 3)
        dilation = args.get('dilation', 1)
        pad_type = args.get('pad_type', '')
        stride = args.get('stride', 1)

        in_chs = args.get('in_chs', planes)
        out_chs = args.get('out_chs', planes)
        if 'out_chs' in args:
            planes = out_chs

        self.act = nn.ReLU(inplace=True)
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

        # Diffusion coefficients
        if self.constant_Dxy:
            self.cDx = nn.Parameter(torch.tensor(args.get('cDx', 1.0)))
            self.cDy = nn.Parameter(torch.tensor(args.get('cDy', 1.0)))
        else:
            self.convDx = create_conv2d(
                planes, planes, kernel_size=dw_kernel_size,
                stride=1, dilation=dilation, padding=pad_type, depthwise=True
            )
            self.convDy = create_conv2d(
                planes, planes, kernel_size=dw_kernel_size,
                stride=1, dilation=dilation, padding=pad_type, depthwise=True
            )
            self.bnDx = norm_layer(planes)
            self.bnDy = norm_layer(planes)

        # Reaction coefficients
        self.conv_alpha = nn.Conv2d(planes, planes, kernel_size=1)
        self.conv_beta = nn.Conv2d(planes, planes, kernel_size=1)
        self.conv_gamma = nn.Conv2d(planes, planes, kernel_size=1)

        # Adaptive time step.
        #
        # dt_live=False (default, published behaviour): the stack ends in Softplus, and
        # forward() computes `Softplus(.) + dt_min` then clamps to [0, dt_max]. Softplus
        # is >= 0 and equals 0.693 at zero pre-activation, i.e. ~7x the 0.1 ceiling, so
        # the clamp saturates on EVERY pixel from initialisation onward. torch.clamp
        # passes zero gradient above its ceiling, so conv_dt has never been trained and
        # dt is effectively a constant.
        #
        # dt_live=True: drop the Softplus and squash with a sigmoid scaled into
        # [dt_min, dt_max]. sigmoid(0) = 0.5 puts dt mid-band at initialisation with
        # full gradient, so the mechanism is live by construction and cannot saturate.
        # This CHANGES the function class -- it is a new arm, not a bug fix to an
        # existing one, and its results are not directly comparable to the default.
        dt_layers = [
            nn.Conv2d(planes, planes, kernel_size=1),
            norm_layer(planes),
            self.act,
            nn.Conv2d(planes, 1, kernel_size=1),
        ]
        if not self.dt_live:
            dt_layers.append(nn.Softplus())
        # Softplus is parameter-free, so the state dict is identical either way and a
        # checkpoint moves between the two modes without renaming anything.
        self.conv_dt = nn.Sequential(*dt_layers)

    def feature_info(self, location):
        return dict(module='', hook_type='', num_chs=self.planes)

    def forward(self, s0, s1=None, drop_path=None):
        f = s1 if self.block_type == 'DartCell' else s0
        h = self.init_h(s0, s1, drop_path) if self.block_type == 'DartCell' else self.init_h(f)

        if (self.stride != 1) or (self.in_chs != self.out_chs):
            f = h
        residual = f

        h_prev = h

        # Diffusion coefficients
        if self.constant_Dxy:
            Dx = torch.abs(self.cDx) + 1e-6
            Dy = torch.abs(self.cDy) + 1e-6
        else:
            Dx = self.act(self.bnDx(self.convDx(h))) + 1e-6
            Dy = self.act(self.bnDy(self.convDy(h))) + 1e-6

        # Reaction coefficients
        alpha = self.conv_alpha(h)
        beta = self.conv_beta(h)
        gamma = self.conv_gamma(h)

        source = f if not self.no_f else 0.0
        if self.dt_live:
            # Bounded into [dt_min, dt_max] by construction, mid-band at init.
            dt_map = self.dt_min + (self.dt_max - self.dt_min) * torch.sigmoid(self.conv_dt(h))
            # CFL guard. Explicit Euler on the 5-point Laplacian needs
            # dt <= dx^2 / (2(Dx+Dy)); with dt no longer pinned at 0.1 this can bite,
            # since cDx/cDy are unconstrained learnables. Scalar under
            # constant_Dxy, so this ceiling does not kill per-pixel gradients the way a
            # per-pixel clamp would.
            cfl = 0.9 * (self.dx * self.dy) / (2.0 * (Dx + Dy) + 1e-12)
            dt_map = torch.minimum(dt_map, cfl.to(dt_map.dtype))
        else:
            dt_map = self.conv_dt(h) + self.dt_min

        for _ in range(self.K):
            diffusion_x = Dx * (
                torch.roll(h_prev, 1, dims=2)
                - 2 * h_prev
                + torch.roll(h_prev, -1, dims=2)
            ) / (self.dx * self.dx)

            diffusion_y = Dy * (
                torch.roll(h_prev, 1, dims=3)
                - 2 * h_prev
                + torch.roll(h_prev, -1, dims=3)
            ) / (self.dy * self.dy)

            diffusion = diffusion_x + diffusion_y

            reaction = alpha * h_prev - beta * h_prev * h_prev + gamma * source

            # The published path re-clamps every iteration (a no-op after the first).
            # Skipped when dt_live, where dt is already bounded by the sigmoid and the
            # CFL ceiling -- re-clamping there would reintroduce the zero-gradient zone
            # this mode exists to remove.
            if not self.dt_live:
                dt_map = torch.clamp(dt_map, 0.0, self.dt_max)

            h_prev = h_prev + dt_map * (diffusion + reaction)

            # hard safety clamp
            h_prev = torch.nan_to_num(h_prev, nan=0.0, posinf=1.0, neginf=0.0)
            h_prev = h_prev.clamp(-5.0, 5.0)


        h = self.bn_out(h_prev)
        h = self.act(h)

        if self.use_res:
            if self.drop_path_rate > 0.:
                h = drop_path(h, self.drop_path_rate, self.training)
            h = h + residual

        return h

