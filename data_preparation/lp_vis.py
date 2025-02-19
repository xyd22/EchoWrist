'''
Low-pass filtering onn the signal to find the clapping gesture
2/21/2022, Ruidong Zhang, rz379@cornell.edu
'''

import os
import cv2
import json
import argparse
import numpy as np
from load_audio import load_audio
from filters import butter_lowpass_filter
from utils import load_default_audio_config
from plot_profiles import plot_profiles_split_channels

def lp_vis(audio_file, audio_config, cutof_time, cutoff_freq=15000):
    _, raw_audio = load_audio(audio_file, audio_config['sample_depth'], 16)
    n_channels = audio_config['n_channels']
    frame_length = audio_config['frame_length']
    raw_audio = np.reshape(raw_audio, (-1, n_channels))

    cuttof_pos = round(cutof_time * audio_config['sampling_rate'] // frame_length) * frame_length
    all_audio = raw_audio[:cuttof_pos]

    for c in range(n_channels):
        all_audio[:, c] = butter_lowpass_filter(all_audio[:, c], cutoff_freq, audio_config['sampling_rate'])

    # all_audio = np.reshape(all_audio[:, 0], (-1, frame_length))      # c x n_frames x frame_length
    all_audio = np.reshape(all_audio.T, (all_audio.shape[1], -1, frame_length))      # c x n_frames x frame_length
    all_audio = np.concatenate(np.swapaxes(all_audio, 1, 2))                  # c x n_frames x frame_length -> c x frame_length x n_frames -> (c x frame_length) x n_frames

    frame_sum = np.sum(np.abs(all_audio), axis=0)
    max_idx = np.argmax(frame_sum)
    syncing_pos = max_idx + 1

    if len(audio_config['channels_of_interest']):
        coi = audio_config['channels_of_interest']
    else:
        coi = list(range(n_channels))
    for c in coi:
        print('File %s Channel %d: signal max: %.1f, after clapping: %.1f, mean: %.1f' % (os.path.basename(audio_file), c, np.max(np.abs(raw_audio[:, c])), np.max(np.abs(raw_audio[(syncing_pos + 5) * frame_length:, c])), np.mean(np.abs(raw_audio[:-5 * audio_config['sampling_rate'], c]))))
    # print('%s, detected syncing pos: %d' % (os.path.basename(audio_file), max_idx + 1))

    hm = plot_profiles_split_channels(np.abs(all_audio), n_channels, None, None)
    cv2.imwrite(audio_file[:-4] + '_lp_vis.png', hm)
    return syncing_pos


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Echo profile calculation')
    parser.add_argument('-a', '--audio', help='path to the audio file, .wav or .raw')
    parser.add_argument('-c', '--config', help='path to the config.json file', default='')
    parser.add_argument('--cutoff_time', help='cutoff time for visualization (s)', type=float, default=15)
    parser.add_argument('--cutoff_freq', help='cutoff frequency for low-pass filter', type=float, default=15000)

    parser.add_argument('--sampling_rate', help='sampling rate of audio file', type=float, default=0)
    parser.add_argument('--n_channels', help='number of channels in audio file', type=int, default=0)
    parser.add_argument('--frame_length', help='length of each audio frame', type=int, default=0)
    parser.add_argument('--sample_depth', help='sampling depth (bit) of audio file', type=int, default=0)

    args = parser.parse_args()

    if len(args.config):
        audio_config = json.load(open(args.config, 'rt'))
        audio_config = audio_config['audio']['config']
    else:
        audio_config = load_default_audio_config()

    if args.sampling_rate > 0:
        audio_config['sampling_rate'] = args.sampling_rate
    if args.n_channels > 0:
        audio_config['n_channels'] = args.n_channels
    if args.frame_length > 0:
        audio_config['frame_length'] = args.frame_length
    if args.sample_depth > 0:
        audio_config['sample_depth'] = args.sample_depth
    
    syncing_pos = lp_vis(args.audio, audio_config, args.cutoff_time, args.cutoff_freq)
    print('%s, detected syncing pos: %d' % (os.path.basename(args.audio), syncing_pos))
