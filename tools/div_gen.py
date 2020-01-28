import os
import ast
import sys
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from got10k.datasets import GOT10k, OTB
from got10k.utils.metrics import rect_iou

sys.path.append('..')
from tracker import *

class ExperimentFR(object):
    def __init__(self, root_dir, subset, report_dir, list_file=None, benchmark='all'):
        self.subset = subset
        self.names = ['GOT-10k', 'OTB2015']
        self.datasets = [GOT10k(os.path.join(root_dir, self.names[0]), subset=subset),
        OTB(os.path.join(root_dir, self.names[1]), 2015)]
        self.report_dir = os.path.expanduser(report_dir)
        self.theta = .4

        if benchmark == self.names[0]:
            self.datasets = [self.datasets[0]]
            self.names = [self.names[0]]
        elif benchmark == self.names[1]:
            self.datasets = [self.datasets[1]]
            self.names = [self.names[1]]

        self.colors = ['r', 'y']
        self.cd = 50

    def run(self, trackers):

        os.makedirs(self.report_dir)
        curr = self.report_dir

        cooldown = 0

        for d, dataset in enumerate(self.datasets):
            for s, (img_files, anno) in enumerate(dataset):
                seq_name = dataset.seq_names[s]

                frame_num = len(img_files)
                boxes = np.zeros((len(trackers), frame_num, 4))

                for f, img_file in enumerate(img_files):
                    cooldown -= 1
                    image = Image.open(img_file)
                    if not image.mode == 'RGB':
                        image = image.convert('RGB')

                    if f == 0:
                        for tracker in trackers:
                            tracker.init(image, anno[0, :])
                    else:
                        for i, tracker in enumerate(trackers):
                            boxes[i, f, :] = tracker.update(image)


                    div = rect_iou(np.array([boxes[0, f, :]]), np.array([boxes[1, f, :]]))

                    if div[0] < self.theta and f > 0 and cooldown <= 0:
                        _, ax = plt.subplots()
                        ax.imshow(image)
                        for i, _ in enumerate(trackers):
                            box = boxes[i, f, :]
                            rect = patches.Rectangle(box[:2], *box[2:],
                                linewidth=1, edgecolor=self.colors[i], facecolor='none')
                            ax.add_patch(rect)
                        rect = patches.Rectangle(anno[f, :][:2], *anno[f, :][2:],
                            linewidth=1, edgecolor='springgreen', facecolor='none')
                        ax.add_patch(rect)
                        name = f"s{s}_f{f}.png"
                        plt.axis('off')
                        plt.savefig(os.path.join(curr, name), bbox_inches='tight', pad_inches=0)
                        cooldown = self.cd

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', help='path to dataset', type=str)
    p.add_argument('--reports', help='path to save reports to', type=str)
    p.add_argument('--models', help='model location', type=str),
    p.add_argument('--benchmark', help='which benchmark', type=str,
                   default='all')
    p.add_argument('--subset', help='subset to test on', type=str)
    args = p.parse_args(sys.argv[1:])

    paths = [os.path.join(args.models, 'dssiam_n2_e50.pth'),
        os.path.join(args.models, 'siamfc_abl_e50.pth')]

    trackers = []
    trackers.append(TrackerSiamFC(backbone=DSSiam(n=1), net_path=paths[0]))
    trackers.append(TrackerSiamFC(backbone=SiamFC(), net_path=paths[1]))

    exp = ExperimentFR(root_dir=args.root, subset=args.subset,
                       report_dir=args.reports, benchmark=args.benchmark)

    exp.run(trackers)
