'''
Automatically detect the clapping positions in .raw audio files
6/26/2022, Ruidong Zhang, rz379@cornell.edu
'''

import os
import json
import argparse

from lp_vis import lp_vis


def audio_auto_sync(parent_path, cutoff_time=15, cutoff_freq=15000):
    config_path = os.path.join(parent_path, 'config.json')
    config = json.load(open(config_path, 'rt'))

    if len(config['audio']['files']):
        all_raw_files = config['audio']['files']
    else:
        all_raw_files = [x for x in os.listdir(parent_path) if x[-4:].lower() == '.raw']
        all_raw_files.sort()
        config['audio']['files'] = all_raw_files

    config['audio']['syncing_poses'] = []

    audio_config = config['audio']['config']

    for f in all_raw_files:
        config['audio']['syncing_poses'] += [int(lp_vis(os.path.join(parent_path, f), audio_config, cutoff_time, cutoff_freq))]

    json.dump(config, open(config_path, 'wt'), indent=4)
    for filename, syncing_pos in zip(config['audio']['files'], config['audio']['syncing_poses']):
        print('Detected syncing pos for %s: %d' % (filename, syncing_pos))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Automatic audio start detection')
    parser.add_argument('-p', '--path', help='path to the folder containing config and .raw audio files')
    parser.add_argument('--cutoff_time', help='cutoff time for visualization (s)', type=float, default=15)
    parser.add_argument('--cutoff_freq', help='cutoff frequency for low-pass filter', type=float, default=15000)

    args = parser.parse_args()

    audio_auto_sync(args.path, args.cutoff_time, args.cutoff_freq)