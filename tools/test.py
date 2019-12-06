import os
import sys
import argparse

from got10k.experiments import *

sys.path.append('..')
from tracker import *


if __name__ == '__main__':

    # handle commandline arguments
    p = argparse.ArgumentParser()
    p.add_argument('--model', help='model to test with', type=str)
    p.add_argument('--weights', help='weights to use for model', type=str)
    p.add_argument('--root', help='path to dataset', type=str)
    p.add_argument('--results', help='path to save results to', type=str)
    p.add_argument('--reports', help='path to save reports to', type=str)
    p.add_argument('--subset', help='subset to test on', type=str)
    args = p.parse_args(sys.argv[1:])

    # setup tracker
    net_path = os.path.expanduser(args.weights)

    if args.model == 'dssiam':
        tracker = TrackerSiamFC(backbone=DSSiam(n=1), net_path=net_path)
    elif args.model == 'siamfc':
        tracker = TrackerSiamFC(backbone=SiamFC(), net_path=net_path)

    tracker.name = args.weights.split('/')[-1]

    # setup experiments
    experiments = [
        ExperimentGOT10k(os.path.expanduser(args.root),
            subset=args.subset,
            result_dir=os.path.expanduser(args.results),
            report_dir=os.path.expanduser(args.reports)
            )
    ]

    # run tracking experiments and report performance
    for e in experiments:
        e.run(tracker)
        e.report([tracker.name])
