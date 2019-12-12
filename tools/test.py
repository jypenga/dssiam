import os
import sys
import argparse

# from got10k.experiments import *

sys.path.append('..')
from tracker import *
from loaders import ExperimentGOT10k, ExperimentOTB


if __name__ == '__main__':

    # handle commandline arguments
    p = argparse.ArgumentParser()
    p.add_argument('--model', help='model to test with', type=str)
    p.add_argument('--weights', help='weights to use for model', type=str)
    p.add_argument('--root', help='path to dataset', type=str)
    p.add_argument('--results', help='path to save results to', type=str)
    p.add_argument('--reports', help='path to save reports to', type=str)
    p.add_argument('--benchmark', help='which benchmark', type=str,
                   default='all')
    p.add_argument('--subset', help='subset to test on', type=str)
    args = p.parse_args(sys.argv[1:])

    # setup tracker
    net_path = os.path.expanduser(args.weights)

    if args.model == 'dssiam':
        tracker = TrackerSiamFC(backbone=DSSiam(n=1), net_path=net_path)
    elif args.model == 'siamfc':
        tracker = TrackerSiamFC(backbone=SiamFC(), net_path=net_path)

    if args.subset == 'val':
        tracker.name = args.weights.split('/')[-1].split('.')[0]
    elif args.subset == 'test':
        tracker.name = args.weights.split('/')[-1].split('.')[0] + '_test'

    # setup experiments
    names = ['GOT-10k', 'OTB2015']
    experiments = [
        ExperimentGOT10k(os.path.join(os.path.expanduser(args.root), names[0]),
            subset=args.subset,
            result_dir=os.path.expanduser(args.results),
            report_dir=os.path.expanduser(args.reports)
            ),
        ExperimentOTB(os.path.join(os.path.expanduser(args.root), names[1]),
            result_dir=os.path.expanduser(args.results),
            report_dir=os.path.expanduser(args.reports)
            )
    ]

    if args.benchmark == 'GOT-10k':
        names == [names[0]]
    if args.benchmark == 'OTB2015':
        names == [names[1]]

    # run tracking experiments and report performance
    for i, name in enumerate(names):
        experiment = experiments[i]
        experiment.run(tracker)
        experiment.report([tracker.name])
