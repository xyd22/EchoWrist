'''
Some utils for data processing
2/17/2022, Ruidong Zhang, rz379@cornell.edu
'''

import os
import re
import json
import warnings

def load_default_audio_config():
    default_audio_config = {
        'sampling_rate': 50000,
        'n_channels': 2,
        'channels_of_interest': [],
        'signal': 'FMCW',
        "tx_file": "fmcw19k.wav",
        'frame_length': 600,
        'sample_depth': 16,
        'bandpass_range': [19000, 23000]
    }
    return default_audio_config

def display_progress(cur_progress, total, update_interval=1000):
    if cur_progress % update_interval == 0 or cur_progress == total - 1:
        print('Progress %d/%d %.2f%%' % (cur_progress + 1, total, 100 * (cur_progress + 1) / total), end='\r')
    if cur_progress == total - 1:
        print('Done.   ')

def load_frame_time(frame_time_file):
    frame_times = []
    with open(frame_time_file, 'rt') as f:
        for l in f.readlines():
            if l[0] > '9' and l[0] < '0':
                continue
            frame_times += [float(l)]

    return frame_times

def extract_labels(loaded_gt):
    labels = {}
    n_cls = max([max([int(y) for y in x[0].split()]) for x in loaded_gt]) + 1
    # print(n_cls)
    for x in loaded_gt:
        for xx, yy in zip(x[0].split(' '), x[3].split(' ')):
            labels[int(xx)] = yy
            if len(labels) == n_cls:
                break
    labels_ordered = [labels[x] if x in labels else '' for x in range(n_cls)]
    # print(labels_ordered)
    return labels_ordered

def assemble_text_labels(indices, extracted_labels):
    text_labels = ' '.join([extracted_labels[int(x)] for x in indices.split()])
    return text_labels

def load_config(parent_folder, config_file=''):
    if len(config_file) == 0:
        config_file = os.path.join(parent_folder, 'config.json')
    if not os.path.exists(config_file):
        raise FileNotFoundError('Config file at %s not found' % config_file)
    config = json.load(open(config_file, 'rt'))
    assert(len(config['audio']['files']) == len(config['audio']['syncing_poses']))  # every audio file should have a sync mark
    assert(len(config['ground_truth']['files']) == len(config['sessions']))         # every ground truth file should have its session config
    assert(len(config['audio']['files']) == len(config['ground_truth']['files']))   # at this moment, audio files and gt files should have same amount

    if len(config['audio']['config']['channels_of_interest']) == 0:
        config['audio']['config']['channels_of_interest'] = list(range(config['audio']['config']['n_channels']))

    for f in config['ground_truth']['files']:
        if not os.path.exists(os.path.join(parent_folder, f)):
            raise FileNotFoundError('Ground truth file at %s not found' % f)        # ground truth file must exist

    if 'videos' not in config['ground_truth'] or len(config['ground_truth']['videos']) == 0:
        gt_videos = []
        for f in config['ground_truth']['files']:
            matched_video = re.findall(r'(\w+\d{6})', f)
            if not matched_video:
                warnings.warn('Warning: could not automatically detect video file for ground truth file %s' % f, RuntimeWarning)
                continue
            gt_videos += [matched_video[0] + '.mp4']
        config['ground_truth']['videos'] = gt_videos
    else:
        assert(len(config['ground_truth']['files']) == len(config['ground_truth']['videos']))   # if you want to specify videos manually, make sure that you cover all of them

    if 'tasks' not in config or len(config['tasks']) == 0:
        config['tasks'] = ['classification']    # default task is classification
    return config