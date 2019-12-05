import os
import sys
import torch
import time
import argparse

from torch.utils.data import DataLoader
from got10k.datasets import GOT10k

sys.path.append('..')
from loaders import *
from tracker import *


if __name__ == '__main__':

    # handle commandline arguments
    p = argparse.ArgumentParser()
    p.add_argument('--model', help='model to train with', type=str)
    p.add_argument('--weights', help='additional custom intialization', type=str)
    p.add_argument('--save', help='path to save epochs to', type=str)
    p.add_argument('--root', help='path to dataset', type=str)
    p.add_argument('--batch_size', help='size of batches', type=int)
    p.add_argument('--num_workers', help='number of cpu workers', type=int)
    p.add_argument('--seq_len', help='length of instance sequence', type=int)
    p.add_argument('--seq_n', help='specific sequence to train on', type=int)
    p.add_argument('--epoch_n', help='amount of epochs to train over', type=int)
    args = p.parse_args(sys.argv[1:])

    start = time.time()

    # setup dataset and tracker
    root_dir = os.path.expanduser(args.root)
    seq_dataset = GOT10k(root_dir, subset='train', return_meta=False)

    if args.model == 'siamfc':
        netpath = os.path.expanduser(args.weights)
        tracker = TrackerSiamFC(backbone=SiamFC(), netpath=netpath)
        if not args.seq_n:
            seq_dataset = Pairwise(seq_dataset)
        else:
            seq_dataset = OnePairwise(seq_dataset, seq_n=args.seq_n)
    elif args.model == 'dssiam':
        netpath = os.path.expanduser(args.weights)
        tracker = TrackerSiamFC(backbone=DSSiam(n=args.seq_len), netpath=netpath)
        if not args.seq_n:
            seq_dataset = Sequential(seq_dataset, n=args.seq_len, max_drift=0)
        else:
            seq_dataset = OneSequential(seq_dataset, seq_n=args.seq_n, n=args.seq_len, max_drift=0)

    # setup data loader
    cuda = torch.cuda.is_available()
    loader = DataLoader(
        seq_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=cuda, drop_last=True, num_workers=args.num_workers)

    # path for saving checkpoints
    net_dir = os.path.expanduser(args.save)
    if not os.path.exists(net_dir):
        os.makedirs(net_dir)

    # re-use weight options
    if not args.weights:
        tracker.net.initialize_weights
        prev_epochs = 0
    else:
        prev_epochs = int(args.weights.split('_')[-1].split('.')[0][1:])

    # training loop
    epoch_num = args.epoch_n
    for epoch in range(epoch_num):
        for step, batch in enumerate(loader):
            if args.model == 'siamfc':
                loss = tracker.step(
                    batch, backward=True, update_lr=(step == 0))
            elif args.model == 'dssiam':
                loss = tracker.ds_step(
                    batch, backward=True, update_lr=(step == 0))
            if step % 20 == 0:
                print('Epoch [{}][{}/{}]: Loss: {:.3f} Time: {:.3f} (s)'.format(
                    epoch + 1, step + 1, len(loader), loss, (time.time() - start)))
                sys.stdout.flush()

        # save checkpoint
        if args.model == 'siamfc':
            name = 'siamfc'
        elif args.model == 'dssiam':
            name = 'dssiam' + '_n' + str(args.seq_len)

        if args.seq_n:
            seq_n = '_' + str(args.seq_n)
        if not args.seq_n:
            seq_n = ''

        net_path = os.path.join(net_dir, name + seq_n + '_e%d.pth' % (epoch + 1))
        torch.save(tracker.net.state_dict(), net_path)

    print('Total time:', time.time() - start)
