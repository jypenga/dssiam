from __future__ import absolute_import, division, print_function

import os
import numpy as np
import glob
import ast
import json
import time
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import cv2

from got10k.datasets import GOT10k, OTB
from got10k.utils.metrics import rect_iou, center_error
from got10k.utils.viz import show_frame
from got10k.utils.ioutils import compress


class ExperimentGOT10k(object):
    r"""Experiment pipeline and evaluation toolkit for GOT-10k dataset.

    Args:
        root_dir (string): Root directory of GOT-10k dataset where
            ``train``, ``val`` and ``test`` folders exist.
        subset (string): Specify ``train``, ``val`` or ``test``
            subset of GOT-10k.
        list_file (string, optional): If provided, only run experiments on
            sequences specified by this file.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """
    def __init__(self, root_dir, subset='val',
                 result_dir='results', report_dir='reports', use_dataset=True):
        super(ExperimentGOT10k, self).__init__()
        assert subset in ['val', 'test']
        self.subset = subset
        if use_dataset:
            self.dataset = GOT10k(
                root_dir, subset=subset)
        self.result_dir = os.path.join(result_dir, 'GOT-10k')
        self.report_dir = os.path.join(report_dir, 'GOT-10k')
        self.nbins_iou = 101
        self.repetitions = 3

    def run(self, tracker, visualize=False, save_video=False):
        if self.subset == 'test':
            print('\033[93m[WARNING]:\n' \
                  'The groundtruths of GOT-10k\'s test set is withholded.\n' \
                  'You will have to submit your results to\n' \
                  '[http://got-10k.aitestunion.com/]' \
                  '\nto access the performance.\033[0m')
            time.sleep(2)

        print('Running tracker %s on GOT-10k...' % tracker.name)
        self.dataset.return_meta = False

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (
                s + 1, len(self.dataset), seq_name))

            # run multiple repetitions for each sequence
            for r in range(self.repetitions):
                # check if the tracker is deterministic
                if r > 0 and tracker.is_deterministic:
                    break
                elif r == 3 and self._check_deterministic(
                    tracker.name, seq_name):
                    print('  Detected a deterministic tracker, ' +
                          'skipping remaining trials.')
                    break
                print(' Repetition: %d' % (r + 1))

                # skip if results exist
                record_file = os.path.join(
                    self.result_dir, tracker.name, seq_name,
                    '%s_%03d.txt' % (seq_name, r + 1))
                if os.path.exists(record_file):
                    print('  Found results, skipping', seq_name)
                    continue

                # tracking loop
                boxes, times = tracker.track(
                    img_files, anno[0, :], visualize=visualize)

                # record results
                self._record(record_file, boxes, times)

            # save videos
            if save_video:
                video_dir = os.path.join(os.path.dirname(os.path.dirname(self.result_dir)),
                    'videos', 'GOT-10k', tracker.name)
                video_file = os.path.join(video_dir, '%s.avi' % seq_name)

                if not os.path.isdir(video_dir):
                    os.makedirs(video_dir)
                image = Image.open(img_files[0])
                img_W, img_H = image.size
                out_video = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'MJPG'), 10, (img_W, img_H))
                for ith, (img_file, pred) in enumerate(zip(img_files, boxes)):
                    image = Image.open(img_file)
                    if not image.mode == 'RGB':
                        image = image.convert('RGB')
                    img = np.array(image)[:, :, ::-1].copy()
                    pred = pred.astype(int)
                    cv2.rectangle(img, (pred[0], pred[1]), (pred[0] + pred[2], pred[1] + pred[3]), self.color['pred'], 2)
                    if ith < anno.shape[0]:
                        gt = anno[ith].astype(int)
                        cv2.rectangle(img, (gt[0], gt[1]), (gt[0] + gt[2], gt[1] + gt[3]), self.color['gt'], 2)
                    out_video.write(img)
                out_video.release()
                print('  Videos saved at', video_file)

    def report(self, tracker_names, plot_curves=True):
        assert isinstance(tracker_names, (list, tuple))

        if self.subset == 'test':
            pwd = os.getcwd()

            # generate compressed submission file for each tracker
            for tracker_name in tracker_names:
                # compress all tracking results
                result_dir = os.path.join(self.result_dir, tracker_name)
                os.chdir(result_dir)
                save_file = '../%s' % tracker_name
                compress('.', save_file)
                print('Records saved at', save_file + '.zip')

            # print submission guides
            print('\033[93mLogin and follow instructions on')
            print('http://got-10k.aitestunion.com/submit_instructions')
            print('to upload and evaluate your tracking results\033[0m')

            # switch back to previous working directory
            os.chdir(pwd)

            return None
        elif self.subset == 'val':
            # meta information is useful when evaluation
            self.dataset.return_meta = True

            # assume tracker_names[0] is your tracker
            report_dir = os.path.join(self.report_dir, tracker_names[0])
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            report_file = os.path.join(report_dir, 'performance.json')

            # visible ratios of all sequences
            seq_names = self.dataset.seq_names
            covers = {s: self.dataset[s][2]['cover'][1:] for s in seq_names}

            performance = {}
            for name in tracker_names:
                print('Evaluating', name)
                ious = {}
                ces = {}
                times = {}
                performance.update({name: {
                    'overall': {},
                    'seq_wise': {}}})

                for s, (_, anno, meta) in enumerate(self.dataset):
                    seq_name = self.dataset.seq_names[s]
                    record_files = glob.glob(os.path.join(
                        self.result_dir, name, seq_name,
                        '%s_[0-9]*.txt' % seq_name))
                    if len(record_files) == 0:
                        raise Exception('Results for sequence %s not found.' % seq_name)

                    # read results of all repetitions
                    boxes = [np.loadtxt(f, delimiter=',') for f in record_files]
                    assert all([b.shape == anno.shape for b in boxes])

                    # calculate and stack all ious
                    bound = ast.literal_eval(meta['resolution'])
                    seq_ious = [rect_iou(b[1:], anno[1:], bound=bound) for b in boxes]
                    seq_ces = [center_error(b[1:], anno[1:]) for b in boxes]
                    # only consider valid frames where targets are visible
                    seq_ious = [t[covers[seq_name] > 0] for t in seq_ious]
                    seq_ces = [t[covers[seq_name] > 0] for t in seq_ces]
                    seq_ious = np.concatenate(seq_ious)
                    seq_ces = np.concatenate(seq_ces)
                    ious[seq_name] = seq_ious
                    ces[seq_name] = seq_ces

                    # stack all tracking times
                    times[seq_name] = []
                    time_file = os.path.join(
                        self.result_dir, name, seq_name,
                        '%s_time.txt' % seq_name)
                    if os.path.exists(time_file):
                        seq_times = np.loadtxt(time_file, delimiter=',')
                        seq_times = seq_times[~np.isnan(seq_times)]
                        seq_times = seq_times[seq_times > 0]
                        if len(seq_times) > 0:
                            times[seq_name] = seq_times

                    # store sequence-wise performance
                    ao, sr, ce, r, speed, _ = self._evaluate(seq_ious, seq_ces, seq_times)
                    performance[name]['seq_wise'].update({seq_name: {
                        'ao': ao,
                        'sr': sr,
                        'ce': ce,
                        'r': r,
                        'speed_fps': speed,
                        'length': len(anno) - 1}})

                ious = np.concatenate(list(ious.values()))
                ces = np.concatenate(list(ces.values()))
                times = np.concatenate(list(times.values()))

                # store overall performance
                ao, sr, ce, r, speed, succ_curve = self._evaluate(ious, ces, times)
                performance[name].update({'overall': {
                    'ao': ao,
                    'sr': sr,
                    'ce': ce,
                    'r': r,
                    'speed_fps': speed,
                    'succ_curve': succ_curve.tolist()}})

            # save performance
            with open(report_file, 'w') as f:
                json.dump(performance, f, indent=4)
            # plot success curves
            if plot_curves:
                self.plot_curves([report_file], tracker_names)

            return performance

    def show(self, tracker_names, seq_names=None, play_speed=1):
        if seq_names is None:
            seq_names = self.dataset.seq_names
        elif isinstance(seq_names, str):
            seq_names = [seq_names]
        assert isinstance(tracker_names, (list, tuple))
        assert isinstance(seq_names, (list, tuple))

        play_speed = int(round(play_speed))
        assert play_speed > 0
        self.dataset.return_meta = False

        for s, seq_name in enumerate(seq_names):
            print('[%d/%d] Showing results on %s...' % (
                s + 1, len(seq_names), seq_name))

            # load all tracking results
            records = {}
            for name in tracker_names:
                record_file = os.path.join(
                    self.result_dir, name, seq_name,
                    '%s_001.txt' % seq_name)
                records[name] = np.loadtxt(record_file, delimiter=',')

            # loop over the sequence and display results
            img_files, anno = self.dataset[seq_name]
            for f, img_file in enumerate(img_files):
                if not f % play_speed == 0:
                    continue
                image = Image.open(img_file)
                boxes = [anno[f]] + [
                    records[name][f] for name in tracker_names]
                show_frame(image, boxes,
                           legends=['GroundTruth'] + tracker_names,
                           colors=['w', 'r', 'g', 'b', 'c', 'm', 'y',
                                   'orange', 'purple', 'brown', 'pink'])

    def _record(self, record_file, boxes, times):
        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        while not os.path.exists(record_file):
            print('warning: recording failed, retrying...')
            np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        print('  Results recorded at', record_file)

        # record running times
        time_file = record_file[:record_file.rfind('_')] + '_time.txt'
        times = times[:, np.newaxis]
        if os.path.exists(time_file):
            exist_times = np.loadtxt(time_file, delimiter=',')
            if exist_times.ndim == 1:
                exist_times = exist_times[:, np.newaxis]
            times = np.concatenate((exist_times, times), axis=1)
        np.savetxt(time_file, times, fmt='%.8f', delimiter=',')

    def _check_deterministic(self, tracker_name, seq_name):
        record_dir = os.path.join(
            self.result_dir, tracker_name, seq_name)
        record_files = sorted(glob.glob(os.path.join(
            record_dir, '%s_[0-9]*.txt' % seq_name)))

        if len(record_files) < 3:
            return False

        records = []
        for record_file in record_files:
            with open(record_file, 'r') as f:
                records.append(f.read())

        return len(set(records)) == 1

    def _evaluate(self, ious, ces, times):
        # AO, SR and tracking speed
        ao = np.mean(ious)
        sr = np.mean(ious > 0.5)
        ce = np.mean(ces)
        r = np.mean(ious > 0.05)
        if len(times) > 0:
            # times has to be an array of positive values
            speed_fps = np.mean(1. / times)
        else:
            speed_fps = -1

        # success curve
        # thr_iou = np.linspace(0, 1, 101)
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        bin_iou = np.greater(ious[:, None], thr_iou[None, :])
        succ_curve = np.mean(bin_iou, axis=0)

        return ao, sr, ce, r, speed_fps, succ_curve

    def plot_curves(self, report_files, tracker_names, extension='.png'):
        assert isinstance(report_files, list), \
            'Expected "report_files" to be a list, ' \
            'but got %s instead' % type(report_files)

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        performance = {}
        for report_file in report_files:
            with open(report_file) as f:
                performance.update(json.load(f))

        succ_file = os.path.join(report_dir, 'success_plot'+extension)
        key = 'overall'

        # filter performance by tracker_names
        performance = {k:v for k,v in performance.items() if k in tracker_names}

        # sort trackers by AO
        tracker_names = list(performance.keys())
        aos = [t[key]['ao'] for t in performance.values()]
        inds = np.argsort(aos)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['succ_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (
                name, performance[name][key]['ao']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='lower left',
                           bbox_to_anchor=(0., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots on GOT-10k')
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

class ExperimentOTB(object):
    r"""Experiment pipeline and evaluation toolkit for OTB dataset.

    Args:
        root_dir (string): Root directory of OTB dataset.
        version (integer or string): Specify the benchmark version, specify as one of
            ``2013``, ``2015``, ``tb50`` and ``tb100``. Default is ``2015``.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """
    def __init__(self, root_dir, version=2015,
                 result_dir='results', report_dir='reports'):
        super(ExperimentOTB, self).__init__()
        self.dataset = OTB(root_dir, version, download=True)
        self.result_dir = os.path.join(result_dir, 'OTB' + str(version))
        self.report_dir = os.path.join(report_dir, 'OTB' + str(version))
        # as nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = 21
        self.nbins_ce = 51

    def run(self, tracker, visualize=False):
        print('Running tracker %s on %s...' % (
            tracker.name, type(self.dataset).__name__))

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))

            # skip if results exist
            record_file = os.path.join(
                self.result_dir, tracker.name, '%s.txt' % seq_name)
            if os.path.exists(record_file):
                print('  Found results, skipping', seq_name)
                continue

            # tracking loop
            boxes, times = tracker.track(
                img_files, anno[0, :], visualize=visualize)
            assert len(boxes) == len(anno)

            # record results
            self._record(record_file, boxes, times)

    def report(self, tracker_names, plot_curves=True):
        assert isinstance(tracker_names, (list, tuple))

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
        report_file = os.path.join(report_dir, 'performance.json')

        performance = {}
        for name in tracker_names:
            print('Evaluating', name)
            seq_num = len(self.dataset)
            succ_curve = np.zeros((seq_num, self.nbins_iou))
            prec_curve = np.zeros((seq_num, self.nbins_ce))
            speeds = np.zeros(seq_num)
            ces = np.zeros(seq_num) # custom

            performance.update({name: {
                'overall': {},
                'seq_wise': {}}})

            for s, (_, anno) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]
                record_file = os.path.join(
                    self.result_dir, name, '%s.txt' % seq_name)
                boxes = np.loadtxt(record_file, delimiter=',')
                boxes[0] = anno[0]
                if not (len(boxes) == len(anno)):
                    print('warning: %s anno donnot match boxes'%seq_name)
                    len_min = min(len(boxes),len(anno))
                    boxes = boxes[:len_min]
                    anno = anno[:len_min]
                assert len(boxes) == len(anno)

                ious, center_errors = self._calc_metrics(boxes, anno)
                ces[s] = np.mean(center_errors) # custom
                succ_curve[s], prec_curve[s] = self._calc_curves(ious, center_errors)

                # calculate average tracking speed
                time_file = os.path.join(
                    self.result_dir, name, 'times/%s_time.txt' % seq_name)
                if os.path.isfile(time_file):
                    times = np.loadtxt(time_file)
                    times = times[times > 0]
                    if len(times) > 0:
                        speeds[s] = np.mean(1. / times)

                # store sequence-wise performance
                performance[name]['seq_wise'].update({seq_name: {
                    'success_curve': succ_curve[s].tolist(),
                    'precision_curve': prec_curve[s].tolist(),
                    'success_score': np.mean(succ_curve[s]),
                    'precision_score': prec_curve[s][20],
                    'success_rate': succ_curve[s][self.nbins_iou // 2],
                    'speed_fps': speeds[s] if speeds[s] > 0 else -1}})

            succ_curve = np.mean(succ_curve, axis=0)
            prec_curve = np.mean(prec_curve, axis=0)
            succ_score = np.mean(succ_curve)
            cent_err = np.mean(ces) # custom
            prec_score = prec_curve[20]
            succ_rate = succ_curve[self.nbins_iou // 2]
            if np.count_nonzero(speeds) > 0:
                avg_speed = np.sum(speeds) / np.count_nonzero(speeds)
            else:
                avg_speed = -1

            # store overall performance
            performance[name]['overall'].update({
                'center_error': cent_err,
                'success_curve': succ_curve.tolist(),
                'precision_curve': prec_curve.tolist(),
                'success_score': succ_score,
                'precision_score': prec_score,
                'success_rate': succ_rate,
                'speed_fps': avg_speed})

        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        # plot precision and success curves
        if plot_curves:
            self.plot_curves(tracker_names)

        return performance

    def show(self, tracker_names, seq_names=None, play_speed=1):
        if seq_names is None:
            seq_names = self.dataset.seq_names
        elif isinstance(seq_names, str):
            seq_names = [seq_names]
        assert isinstance(tracker_names, (list, tuple))
        assert isinstance(seq_names, (list, tuple))

        play_speed = int(round(play_speed))
        assert play_speed > 0

        for s, seq_name in enumerate(seq_names):
            print('[%d/%d] Showing results on %s...' % (
                s + 1, len(seq_names), seq_name))

            # load all tracking results
            records = {}
            for name in tracker_names:
                record_file = os.path.join(
                    self.result_dir, name, '%s.txt' % seq_name)
                records[name] = np.loadtxt(record_file, delimiter=',')

            # loop over the sequence and display results
            img_files, anno = self.dataset[seq_name]
            for f, img_file in enumerate(img_files):
                if not f % play_speed == 0:
                    continue
                image = Image.open(img_file)
                boxes = [anno[f]] + [
                    records[name][f] for name in tracker_names]
                show_frame(image, boxes,
                           legends=['GroundTruth'] + tracker_names,
                           colors=['w', 'r', 'g', 'b', 'c', 'm', 'y',
                                   'orange', 'purple', 'brown', 'pink'])

    def _record(self, record_file, boxes, times):
        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        while not os.path.exists(record_file):
            print('warning: recording failed, retrying...')
            np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        print('  Results recorded at', record_file)

        # record running times
        time_dir = os.path.join(record_dir, 'times')
        if not os.path.isdir(time_dir):
            os.makedirs(time_dir)
        time_file = os.path.join(time_dir, os.path.basename(
            record_file).replace('.txt', '_time.txt'))
        np.savetxt(time_file, times, fmt='%.8f')

    def _calc_metrics(self, boxes, anno):
        # can be modified by children classes
        ious = rect_iou(boxes, anno)
        center_errors = center_error(boxes, anno)
        return ious, center_errors

    def _calc_curves(self, ious, center_errors):
        ious = np.asarray(ious, float)[:, np.newaxis]
        center_errors = np.asarray(center_errors, float)[:, np.newaxis]

        thr_iou = np.linspace(0, 1, self.nbins_iou)[np.newaxis, :]
        thr_ce = np.arange(0, self.nbins_ce)[np.newaxis, :]

        bin_iou = np.greater(ious, thr_iou)
        bin_ce = np.less_equal(center_errors, thr_ce)

        succ_curve = np.mean(bin_iou, axis=0)
        prec_curve = np.mean(bin_ce, axis=0)

        return succ_curve, prec_curve

    def plot_curves(self, tracker_names):
        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        assert os.path.exists(report_dir), \
            'No reports found. Run "report" first' \
            'before plotting curves.'
        report_file = os.path.join(report_dir, 'performance.json')
        assert os.path.exists(report_file), \
            'No reports found. Run "report" first' \
            'before plotting curves.'

        # load pre-computed performance
        with open(report_file) as f:
            performance = json.load(f)

        succ_file = os.path.join(report_dir, 'success_plots.png')
        prec_file = os.path.join(report_dir, 'precision_plots.png')
        key = 'overall'

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # sort trackers by success score
        tracker_names = list(performance.keys())
        succ = [t[key]['success_score'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['success_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['success_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='center left',
                           bbox_to_anchor=(1, 0.5))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots of OPE')
        ax.grid(True)
        fig.tight_layout()

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

        # sort trackers by precision score
        tracker_names = list(performance.keys())
        prec = [t[key]['precision_score'] for t in performance.values()]
        inds = np.argsort(prec)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot precision curves
        thr_ce = np.arange(0, self.nbins_ce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_ce,
                            performance[name][key]['precision_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['precision_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='center left',
                           bbox_to_anchor=(1, 0.5))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Location error threshold',
               ylabel='Precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1),
               title='Precision plots of OPE')
        ax.grid(True)
        fig.tight_layout()

        print('Saving precision plots to', prec_file)
        fig.savefig(prec_file, dpi=300)
