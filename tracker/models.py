from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .modules import AffineCentering, SoftArgmax2D


class DSSiam(nn.Module):
    """Non-debug version."""
    def __init__(self, n=1), reg=False):
        super(DSSiam, self).__init__()
        self.n = n
        self.reg = reg

        # additional modules
        self.SoftArgmax = SoftArgmax2D(upsample=255)
        self.AffineCentering = AffineCentering()
        self.BatchNorm = nn.BatchNorm2d(1, eps=1e-6, momentum=0.05)

        # convolutional stages of AlexNet
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, 11, 2, bias=False),
            nn.BatchNorm2d(96, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(96, 256, 5, 1, groups=2, bias=False),
            nn.BatchNorm2d(256, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(256, 384, 3, 1, bias=False),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(384, 384, 3, 1, groups=2, bias=False),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(384, 256, 3, 1, groups=2))
        self.initialize_weights()

    def forward(self, z, xs, x_cs):
        outs = torch.Tensor().to(z.device)
        features = torch.Tensor().to(z.device)

        # reshape x image batch
        b, s, c, xd, yd = xs.size()
        xs = xs.view(b * s, c, xd, yd)
        num = b * s // self.n

        # rehape x center batch
        b, s, l = x_cs.size()
        center = x_cs.view(b * s, l)

        # setup exemplar
        z = self.feature(z)

        center = center[self._get_indices(num)]

        # feed sequence through n shared layers
        for i in range(self.n):
            x = self.AffineCentering(xs[self._get_indices(num, offset=i)],
                                     center)

            # obtain instance features
            x = self.feature(x)

            features = torch.cat((features, x))

            n, c, h, w = x.size()
            x = x.view(1, n * c, h, w)

            # reshape and normalize responses
            out = F.conv2d(x, z, groups=n)
            out = out.view(n, 1, out.size(-2), out.size(-1))
            out = self.BatchNorm(out)

            # adjust the scale of responses
            outs = torch.cat((outs, (out + 0.0).unsqueeze(0)))

            center = self.SoftArgmax(outs[i] * 1e3).squeeze()

        if self.reg:
            return outs, self._gram_det(features, num)
        else:
            return outs, torch.zeros(batch_size).to(z.device)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, mode='fan_out',
                                     nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)

    def _gram_det(self, features, batch_size):
        ds = torch.zeros(batch_size).to(features.device)
        for i in range(batch_size):
            indices = self._get_indices(self.n, offset=i, incr=batch_size)
            V = torch.flatten(features[indices], start_dim=1).transpose(0, 1)
            G = V.transpose(0, 1) @ V
            ds[i] = torch.norm(G, p='nuc')
        return ds

    def _get_indices(self, n, offset=0, incr=None):
        incr = self.n if not incr else incr
        indices = []
        for i in range(n):
                indices.append(i * incr + offset)
        return indices


class SiamFC(nn.Module):

    def __init__(self):
        super(SiamFC, self).__init__()
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(384, 256, 3, 1, groups=2))
        self.initialize_weights()

    def forward(self, z, x):
        z = self.feature(z)
        x = self.feature(x)

        # fast cross correlation
        n, c, h, w = x.size()
        x = x.view(1, n * c, h, w)
        out = F.conv2d(x, z, groups=n)
        out = out.view(n, 1, out.size(-2), out.size(-1))

        # adjust the scale of responses
        out = 0.001 * out + 0.0

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, mode='fan_out',
                                     nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
