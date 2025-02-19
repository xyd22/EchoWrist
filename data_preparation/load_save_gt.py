'''
Load and parse a ground truth file
2/17/2022, Ruidong Zhang, rz379@cornell.edu
'''

import numpy as np

def load_gt(gt_file):
    if gt_file[-4:] == '.npy':
        return np.load(gt_file)
    gt = []
    with open(gt_file, 'rt') as f:
        for line in f.readlines():
            items = line.strip('\n').split(',')
            gt += [items]
    return gt

def save_gt(gt, target_file):
    if target_file[-4:] == '.npy':
        np.save(target_file, gt)
        return
    with open(target_file, 'wt') as f:
        for r in gt:
            for i, item in enumerate(r):
                f.write(str(item))
                if i < len(r) - 1:
                    f.write(',')
                else:
                    f.write('\n')