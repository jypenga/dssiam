from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class AffineCentering(nn.Module):
    """
    Implementation of the DS-SiamFC pre-processing stage.
    """

    def __init__(self, mode='zeros'):
        super(AffineCentering, self).__init__()
        self.mode = mode

    def forward(self, x, centers):
        # obtain input dimensions
        b, c, yd, xd = x.size()

        # determine translations
        center = torch.Tensor([(xd/2), (yd/2)]).to(x.device)
        ts = -(centers.to(x.device) - center.to(x.device))

        # create affine matrices
        theta = torch.Tensor([[[1., 0., -(t[0]/xd)*2],
                              [0., 1., -(t[1]/yd)*2]]
                              for t in ts]).to(x.device) # align_corners

        # build affine grids with theta
        transform = F.affine_grid(theta,
            torch.Size([b, c, yd, xd]))

        return F.grid_sample(x, transform, padding_mode=self.mode) # align_corners


class SoftArgmax2D(nn.Module):
    """
    Implementation of SoftArgmax2D, adopted from the kornia module.
    Originally SpatialSoftArgmax2d. Source code for original available at
    https://torchgeometry.readthedocs.io/en/latest/_modules/.
    Rewritten to fit this project.
    """

    def __init__(self, upsample=False):
        super(SoftArgmax2D, self).__init__()
        self.eps = 1e-6
        self.ups = upsample

    def _create_meshgrid(self, x):
        _, _, height, width = x.shape
        _dv, _dt = x.device, x.dtype

        xs = torch.linspace(0, width - 1, width, device=_dv, dtype=_dt)
        ys = torch.linspace(0, height - 1, height, device=_dv, dtype=_dt)
        return torch.meshgrid(ys, xs)  # pos_y, pos_x

    def forward(self, x):
        # upsample input tensor
        if self.ups:
            m = nn.Upsample(size=self.ups, mode='bilinear', align_corners=False)
            x = m(x)

        # unpack shapes and create view from input tensor
        batch_size, channels, height, width = x.shape
        z = x.view(batch_size, channels, -1)

        # compute softmax with max substraction trick
        exp_x = torch.exp(z - torch.max(z, dim=-1, keepdim=True)[0])
        exp_x_sum = torch.tensor(
            1.0) / (exp_x.sum(dim=-1, keepdim=True) + self.eps)

        # create coordinates grid
        pos_y, pos_x = self._create_meshgrid(x)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        # compute the expected coordinates
        expected_y = torch.sum((pos_y * exp_x) * exp_x_sum, dim=-1,
            keepdim=True)
        expected_x = torch.sum((pos_x * exp_x) * exp_x_sum, dim=-1,
            keepdim=True)
        output = torch.cat([expected_x, expected_y], dim=-1)
        return output.view(batch_size, channels, 2)  # x, y
