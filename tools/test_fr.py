import os
import ast
import sys
import json
import argparse

import numpy as np

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
        self.report_dir = os.path.join(report_dir, 'FR')
        self.theta = .1

        if benchmark == self.names[0]:
            self.datasets = [self.datasets[0]]
            self.names = [self.names[0]]
        elif benchmark == self.names[1]:
            self.datasets = [self.datasets[1]]
            self.names = [self.names[1]]

        self.dict = {name:{'total':{}, 'seq_wise':{}} for name in self.names}

    def run(self, tracker):

        for d, dataset in enumerate(self.datasets):
            key = self.names[d]
            d_fr = 0

            for s, (img_files, anno) in enumerate(dataset):
                seq_name = dataset.seq_names[s]
                print(seq_name)

                s_fr = 0

                frame_num = len(img_files)
                print(frame_num)
                boxes = np.zeros((frame_num, 4))

                for f, img_file in enumerate(img_files):
                    print(f)
                    image = Image.open(img_file)
                    if not image.mode == 'RGB':
                        image = image.convert('RGB')

                    # init on first frame of s
                    if f == 0:
                        tracker.init(image, anno[0, :])
                    else:
                        boxes[f, :] = tracker.update(image)

                    iou = rect_iou(np.array([boxes[f, :]]), np.array([anno[f, :]]))

                    # re-init if prediction overlap < .1
                    if iou[0] <= self.theta and f > 0:
                        if f < (frame_num - 1):
                            tracker.init(image, anno[f, :])
                        d_fr += 1
                        s_fr += 1

                self.dict[key]['seq_wise'][s] = {'fr':s_fr, 'length':frame_num}
            self.dict[key]['total'] = {'fr':d_fr}

        report_file = os.path.join(self.report_dir, 'fr.json')

        with open(report_file, 'w') as f:
            json.dump(self.dict, f, indent=4)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', help='model to test with', type=str)
    p.add_argument('--weights', help='weights to use for model', type=str)
    p.add_argument('--root', help='path to dataset', type=str)
    p.add_argument('--reports', help='path to save reports to', type=str)
    p.add_argument('--benchmark', help='which benchmark', type=str,
                   default='all')
    p.add_argument('--subset', help='subset to test on', type=str)
    args = p.parse_args(sys.argv[1:])

    net_path = os.path.expanduser(args.weights)

    if args.model == 'dssiam':
        tracker = TrackerSiamFC(backbone=DSSiam(n=1), net_path=net_path)
    elif args.model == 'siamfc':
        tracker = TrackerSiamFC(backbone=SiamFC(), net_path=net_path)

    if args.subset == 'val':
        tracker.name = args.weights.split('/')[-1].split('.')[0]
    elif args.subset == 'test':
        tracker.name = args.weights.split('/')[-1].split('.')[0] + '_test'

    exp = ExperimentFR(root_dir=args.root, subset=args.subset,
                       report_dir=args.reports, benchmark=args.benchmark)

    exp.run(tracker)
