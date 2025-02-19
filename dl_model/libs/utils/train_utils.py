from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import random
import logging
import numpy as np

import torch
import torch.optim as optim


from torch.utils.data.sampler import Sampler


class IntervalSampler(Sampler):
    r"""Samples elements with fixed interval randomly.

    Arguments:
        data_source (Dataset): dataset to sample from
        interval (int): interval between two samples
    """

    def __init__(self, data_source, interval):
        self.data_source = data_source
        self.interval = interval
        self.num_samples = len(self.data_source) // interval

    def __iter__(self):
        n = len(self.data_source)
        s = random.randint(0, self.interval-1)
        idx_list = torch.arange(s, n, self.interval).tolist()
        self.num_samples = len(idx_list)
        # inplace shuffle
        random.shuffle(idx_list)
        return iter(idx_list)

    def __len__(self):
        return self.num_samples


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best,
                    file_folder="./models/", filename='checkpoint.pth.tar'):
    """save checkpoint"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, filename))
    if is_best:
        # skip the optimization state
        if 'optimizer' in state:
            state.pop('optimizer', None)
        if 'scheduler' in state:
            state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


################################################################################
def get_params_lr(model, weight_decay, lr, decoder_ratio=1.0):
    """
    Set lr for different params
    """
    encoder_decay, encoder_no_decay = [], []
    decoder_decay, decoder_no_decay = [], []

    for name, param in model.encoder.named_parameters():
        # ignore the param without grads
        if not param.requires_grad:
            continue
        # no weight decay for BN / conv bias params
        if ("bias" in name) or ("bn" in name):
            encoder_no_decay.append(param)
        else:
            encoder_decay.append(param)

    for name, param in model.decoder.named_parameters():
        # ignore the param without grads
        if not param.requires_grad:
            continue
        # no weight decay for BN / conv bias params
        if ("bias" in name) or ("bn" in name):
            decoder_no_decay.append(param)
        else:
            decoder_decay.append(param)

    return [{'params': encoder_no_decay,
             'weight_decay': 0.,
             'lr': lr},
            {'params': encoder_decay,
             'weight_decay': weight_decay,
             'lr': lr},
            {'params': decoder_no_decay,
             'weight_decay': 0.,
             'lr': decoder_ratio * lr},
            {'params': decoder_decay,
             'weight_decay': weight_decay,
             'lr': decoder_ratio * lr}]


def create_optim(model, optimizer_config):
    """get optimizer
    return a supported optimizer
    """
    params = get_params_lr(model,
                           optimizer_config["weight_decay"],
                           optimizer_config["learning_rate"],
                           decoder_ratio=optimizer_config["decoder_lr_ratio"])
    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(params,
                              lr=optimizer_config["learning_rate"],
                              momentum=optimizer_config["momentum"],
                              nesterov=optimizer_config["nesterov"])
    elif optimizer_config["type"] == "Adam":
        optimizer = optim.Adam(params,
                               lr=optimizer_config["learning_rate"])
    elif optimizer_config["type"] == "RMSprop":
        optimizer = optim.RMSprop(params,
                               lr=optimizer_config["learning_rate"])
    else:
        raise TypeError("Unsupported solver")

    return optimizer


def create_scheduler(optimizer, schedule_config, max_epochs, num_iters,
                     last_epoch=-1):
    if schedule_config["type"] == "cosine":
        # step per iteration
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, num_iters * max_epochs,
            last_epoch=last_epoch)
    elif schedule_config["type"] == "multistep":
        # step every some epochs
        steps = [num_iters * step for step in schedule_config["steps"]]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, steps,
            gamma=schedule_config["gamma"], last_epoch=last_epoch)
    else:
        raise TypeError("Unsupported scheduler")

    return scheduler


def display_progress(cur_progress, total, update_interval=1000):
    if cur_progress % update_interval == 0 or cur_progress == total - 1:
        print('Progress %d/%d %.2f%%' % (cur_progress + 1,
                                         total, 100 * (cur_progress + 1) / total), end='\r')
    if cur_progress == total - 1:
        print('Done.   ')


def print_and_log(content, end='\n'):
    print(content, end=end)
    logging.info(content)


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

def extract_labels(loaded_gt):
    labels = {}
    n_cls = max([int(x[0]) for x in loaded_gt]) + 1
    for x in loaded_gt:
        labels[int(x[0])] = x[3]
        if len(labels) == n_cls:
            break
    labels_ordered = [labels[x] if x in labels else '' for x in range(n_cls)]
    return labels_ordered

def plot_profiles(profiles, max_val=None, min_val=None):
    max_h = 0       # red
    min_h = 120     # blue
    if not max_val:
        max_val = np.max(profiles)
    if not min_val:
        min_val = np.min(profiles)
    print(max_val, min_val)
    heat_map_val = np.clip(profiles, min_val, max_val)
    heat_map = np.zeros(
        (heat_map_val.shape[0], heat_map_val.shape[1], 3), dtype=np.uint8)
    # print(heat_map_val.shape)
    heat_map[:, :, 0] = heat_map_val / \
        (max_val + 1e-6) * (max_h - min_h) + min_h
    heat_map[:, :, 1] = np.ones(heat_map_val.shape) * 255
    heat_map[:, :, 2] = np.ones(heat_map_val.shape) * 255
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_HSV2BGR)
    return heat_map
