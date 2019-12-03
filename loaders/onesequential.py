# !/usr/bin/python
# -*- coding: utf-8 -*-

"""
Dataloader intended for Deeply Supervised SiamFC.
Jorrit Ypenga (11331550) UvA FNWI.
"""


import torch

import numpy as np

from torchvision.transforms import ToTensor, CenterCrop
from torch.utils.data import Dataset
from PIL import Image, ImageStat, ImageOps


class OneSequential(Dataset):

    def __init__(self, seq_dataset, seq_n, n=3, f_diff=1, max_drift=0):
        super(OneSequential, self).__init__()
        self.n = n
        self.f_diff = f_diff
        self.max_drift = max_drift

        self.seq_dataset = seq_dataset
        self.seq_n = seq_n

        # config variables
        self.pairs_per_seq = 10000
        self.max_dist = 100
        self.exemplar_sz = 127
        self.instance_sz = 255
        self.context = .5

    def __getitem__(self, index):
        # obtain image files and annotations
        index = self.seq_n
        img_files, anno = self.seq_dataset[index]
        max_frame = len(img_files) - self.n * self.f_diff

        # sample random (template, instance sequence) pair
        rand_z, rand_xs = self._sample_pair(max_frame)

        exemplar_image = Image.open(img_files[rand_z])
        instance_images = [Image.open(img_files[x]) for x in rand_xs]

        # obtain images and centers from dataset
        exemplar_box = anno[rand_z]
        instance_boxes = anno[rand_xs]

        exemplar_image, center = self._crop_and_resize(exemplar_image,
                                                       exemplar_box)

        # obtain random drift for initial instance image
        rand_coord = None
        if self.max_drift:
            rand_coord = np.random.random_integers(-self.max_drift,
                                                   self.max_drift, 2)

        # crop and resize exemplar and instances
        new_instance_images, centers = [], []
        for i, img in enumerate(instance_images):
            new_img, cent = self._crop_and_resize(img, instance_boxes[0],
                                                  calc_box=instance_boxes[i],
                                                  rand_coord=rand_coord)
            new_instance_images.append(np.asarray(new_img))
            centers.append(cent)
        instance_images = np.array(new_instance_images)
        centers = np.array(centers)

        # to tensors
        exemplar_image = CenterCrop(self.exemplar_sz)(exemplar_image)
        exemplar_image = ToTensor()(exemplar_image).float()
        instance_images = torch.from_numpy(instance_images).permute([0, 3, 1, 2]).float()
        centers = torch.from_numpy(centers).float()

        return exemplar_image, instance_images, centers

    def __len__(self):
        return self.pairs_per_seq #// self.n

    def _sample_pair(self, n):
        assert n > 0
        if n == 1:
            return 0, 0
        elif n == 2:
            return 0, 1
        else:
            max_dist = min(n - 1, self.max_dist)
            rand_dist = np.random.choice(max_dist) + 1
            rand_z = np.random.choice(n - rand_dist)
            rand_x = rand_z + rand_dist

        return rand_z, np.arange(rand_x, rand_x + self.n * self.f_diff,
            (self.n*self.f_diff)/self.n).astype(int)

    def _crop_and_resize(self, image, crop_box, calc_box=None, rand_coord=None):
        # convert box to 0-indexed and center based
        box = np.array([
            crop_box[0] - 1 + (crop_box[2] - 1) / 2,
            crop_box[1] - 1 + (crop_box[3] - 1) / 2,
            crop_box[2], crop_box[3]], dtype=np.float32)
        center, target_sz = box[:2], box[2:]

        # exemplar and search sizes
        context = self.context * np.sum(target_sz)
        z_sz = np.sqrt(np.prod(target_sz + context))
        x_sz = z_sz * self.instance_sz / self.exemplar_sz

        if np.any(rand_coord):
            rand_coord = (rand_coord * (x_sz/300)).astype(int)

        # calculate center of reference box
        if np.any(calc_box):
            calc_box = np.array([
                calc_box[0] - 1 + (calc_box[2] - 1) / 2,
                calc_box[1] - 1 + (calc_box[3] - 1) / 2,
                calc_box[2], calc_box[3]], dtype=np.float32)

            # adjust center if random coordinate is specified
            if np.any(rand_coord):
                calc_center = calc_box[:2] - rand_coord
            else:
                calc_center = calc_box[:2]

        # convert box to corners (0-indexed)
        size = round(x_sz)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # off-center the image using the random coordinate
        if np.any(rand_coord):
            corners[:2] += rand_coord
            corners[2:] += rand_coord

        # pad image if necessary
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.size))
        npad = max(0, int(pads.max()))
        if npad > 0:
            avg_color = ImageStat.Stat(image).mean
            # PIL doesn't support float RGB image
            avg_color = tuple(int(round(c)) for c in avg_color)
            image = ImageOps.expand(image, border=npad, fill=avg_color)

        # crop image patch
        corners = tuple((corners + npad).astype(int))
        patch = image.crop(corners)

        patch_size = patch.size[0]

        # resize to instance_sz
        out_size = (self.instance_sz, self.instance_sz)
        patch = patch.resize(out_size, Image.BILINEAR)

        # return cropped image + center of tracked object
        if np.any(calc_box):
            calc_center = (255/2) - ((center - calc_center) * (self.instance_sz/patch_size))
            return patch, calc_center
        else:
            return patch, np.array([255/2, 255/2])
