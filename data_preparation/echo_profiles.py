'''
Calculate Echo profiles
2/17/2022, Ruidong Zhang, rz379@cornell.edu
'''

import os
import cv2
import json
import argparse
import numpy as np

from load_audio import load_audio
from filters import butter_bandpass_filter
from utils import load_default_audio_config
from plot_profiles import plot_profiles_split_channels, plot_alternating_channels


def check_integrity(raw_seq, pkt_length=240):
    uniformed_seq = raw_seq[:raw_seq.shape[0] - raw_seq.shape[0] % pkt_length]
    bad_pkt = np.sum(np.abs(np.reshape(
        uniformed_seq, (-1, pkt_length))) < 1e-6, axis=1) == pkt_length
    n_bad_pkt = np.sum(bad_pkt)
    if n_bad_pkt:
        print('Bad packets: %d' % n_bad_pkt)
    bad_pkt = np.ones(
        (uniformed_seq.shape[0] // pkt_length, pkt_length), dtype=bool) * bad_pkt[:, None]
    bad_samples = np.ones(raw_seq.shape, dtype=bool)
    bad_samples[:bad_pkt.ravel().shape[0]] = bad_pkt.ravel()
    return bad_samples


def echo_profiles(audio_file, audio_config, target_bitwidth, maxval, maxdiff, mindiff, rectify=False, no_overlapp=False, no_diff=False, coi_suffix='', alternating=''):

    original_audio, all_audio = load_audio(
        audio_file, audio_config['sample_depth'], target_bitwidth)
    bad_samples = check_integrity(original_audio)
    n_channels = audio_config['n_channels']
    all_audio = np.reshape(all_audio, (-1, n_channels))
    bad_samples = np.reshape(bad_samples, (-1, n_channels))

    if 'tx_file' in audio_config:
        tx_files = []
        if isinstance(audio_config['tx_file'], str):
            tx_files = [audio_config['tx_file']]
        else:
            tx_files = audio_config['tx_file']
    elif audio_config['signal'].lower() == 'gsm':
        tx_files = ['gsm.wav']
    elif audio_config['signal'].lower() == 'chirp':
        tx_files = ['chirp.wav']
    elif audio_config['signal'].lower() == 'fmcw':
        tx_files = ['fmcw.wav']
    else:
        raise RuntimeError('Unknown profiling method: %s' %
                           audio_config['signal'])

    tx_signals = []
    for f in tx_files:
        _, this_tx = load_audio(os.path.join('tx_signals', f))
        frame_length = this_tx.shape[0]
        # frame_length must match
        assert (frame_length == audio_config['frame_length'])
        # if frame_length != audio_config['frame_length']:
        #     this_tx = np.concatenate([this_tx, np.zeros(audio_config['frame_length'] - frame_length)])
        #     frame_length = audio_config['frame_length']
        tx_signals += [this_tx]

    if isinstance(audio_config['bandpass_range'][0], int):
        bp_ranges = [audio_config['bandpass_range']]
    else:
        bp_ranges = audio_config['bandpass_range']

    n_tx = len(tx_signals)

    if len(audio_config['channels_of_interest']) == 0:
        audio_config['channels_of_interest'] = list(range(n_channels))
    n_coi = len(audio_config['channels_of_interest'])

    filtered_audio = np.zeros((n_tx, n_coi, all_audio.shape[0]))

    start_profiles = np.zeros((n_tx, n_coi, min(frame_length * 100000, all_audio.shape[0] // (
        frame_length * n_tx * n_coi) * (frame_length * n_tx * n_coi))))

    for n, tx in enumerate(tx_signals):
        for i, c in enumerate(audio_config['channels_of_interest']):
            filtered_audio[n, i] = butter_bandpass_filter(
                all_audio[:, c], bp_ranges[n][0], bp_ranges[n][1], audio_config['sampling_rate'])
            # find the start pos
            if audio_config['signal'].lower() == 'gsm':
                channel_start_profiles = np.convolve(
                    filtered_audio[n, i][:start_profiles.shape[2]], tx, mode='full')
            else:
                channel_start_profiles = np.correlate(
                    filtered_audio[n, i][:start_profiles.shape[2]], tx, mode='full')
            start_profiles[n, i, :] = channel_start_profiles[frame_length - 1:]

    start_profiles.shape = (n_tx, n_coi, -1, frame_length)
    start_profiles.shape = (-1, start_profiles.shape[3])
    start_profiles = np.mean(np.abs(start_profiles), axis=0)

    # start_pos = 47 + 256
    start_pos = np.argmax(start_profiles)
    start_pos = (start_pos + frame_length - frame_length // 2) % frame_length
    print('Detected start pos: %d' % start_pos)

    # return

    filtered_audio = filtered_audio[:, :, start_pos:]
    # c x seq_len = c x (n_frames x frame_length)
    filtered_audio = filtered_audio[:, :, :filtered_audio.shape[2] -
                                    filtered_audio.shape[2] % frame_length]
    # c x n_frames x frame_length
    filtered_audio = np.reshape(
        filtered_audio, (filtered_audio.shape[0], filtered_audio.shape[1], -1, frame_length))

    # TODO: add rectify back
    # if rectify:
    #     for _ in range(5):
    #         abs_audio = np.abs(filtered_audio)
    #         small_half_window = 50
    #         small_window_avgs = np.zeros((n_coi, frame_length))
    #         for cc in range(frame_length):
    #             small_window_start = max(0, cc - small_half_window)
    #             small_window_end = min(frame_length, cc + small_half_window + 1)
    #             small_window_avgs[:, cc] = np.mean(abs_audio[:, :, small_window_start: small_window_end], axis=(1, 2))
    #             # print(small_window_avgs[:, cc])
    #         for c in range(n_coi):
    #             filtered_audio[c, :, :] /= small_window_avgs[c]

    profiles = np.zeros(
        (n_tx, n_coi, filtered_audio.shape[2] * filtered_audio.shape[3]))

    if no_overlapp:
        no_overlapp_text = '_no_overlapp'
        profiles = np.reshape(
            profiles, (n_tx, profiles.shape[0], -1, frame_length))
        for n, tx in enumerate(tx_signals):
            for c in range(n_coi):
                if audio_config['signal'].lower() == 'gsm':
                    for f in range(profiles.shape[1]):
                        profiles[n, c, f, :] = np.convolve(filtered_audio[n, c, f, :], tx, mode='full')[
                            frame_length - 1:]
                else:
                    for f in range(profiles.shape[1]):
                        profiles[n, c, f, :] = np.correlate(
                            filtered_audio[n, c, f, :], tx, mode='full')[frame_length - 1:]
    else:
        no_overlapp_text = ''
        for n, tx in enumerate(tx_signals):
            for c in range(n_coi):
                if audio_config['signal'].lower() == 'gsm':
                    profiles[n, c, :] = np.convolve(filtered_audio[n, c].ravel(), tx, mode='full')[
                        frame_length - 1:]
                else:
                    profiles[n, c, :] = np.correlate(filtered_audio[n, c].ravel(), tx, mode='full')[
                        frame_length - 1:]
        profiles = np.reshape(
            profiles, (n_tx, profiles.shape[1], -1, frame_length))
    # profiles = np.reshape(profiles, (-1, profiles.shape[2], profiles.shape[3]))
    # c x n_frames x frame_length -> c x frame_length x n_frames -> (c x frame_length) x n_frames
    profiles = profiles.swapaxes(2, 3)
    profiles = np.reshape(profiles, (-1, profiles.shape[3]))
    # profiles.shape = -1, profiles.shape[2]

    bad_samples = bad_samples[start_pos:]
    bad_samples = bad_samples[:len(
        bad_samples) - len(bad_samples) % frame_length]
    # if one channel is bad, all channels are bad
    bad_samples = np.reshape(bad_samples[:, 0], (-1, frame_length)).T
    bad_frame = np.zeros(profiles.shape[1], dtype=bool)
    bad_frame[:bad_samples.shape[1]] = (np.sum(bad_samples, axis=0) > 0)

    profiles *= (1 - bad_frame[None, :])
    profiles_img = plot_profiles_split_channels(
        profiles, n_tx * n_coi, maxval, minval=0)
    filename = ('%s_%s%s_%dbit_profiles' % (
        audio_file[:-4], audio_config['signal'].lower(), no_overlapp_text, target_bitwidth)) + coi_suffix
    cv2.imwrite(filename + '.png', profiles_img)
    np.save(filename + '.npy', profiles)
    del profiles_img  # save memory

    if alternating:
        alternating_img = plot_alternating_channels(
            profiles, n_tx*n_coi, maxval, minval=0)
        filename = ('%s_%s%s_%dbit_alternating_profiles' % (
            audio_file[:-4], audio_config['signal'].lower(), no_overlapp_text, target_bitwidth)) + coi_suffix
        cv2.imwrite(filename + '.png', alternating_img)
        np.save(filename + '.npy', profiles)
        del alternating_img

    if not no_diff:
        diff_profiles = np.abs(profiles[:, 1:]) - np.abs(profiles[:, :-1])
        diff_profiles *= (1 - bad_frame[None, 1:]) * (1 - bad_frame[None, :-1])
        diff_profiles_img = plot_profiles_split_channels(
            diff_profiles, n_tx * n_coi, maxdiff, mindiff)
        filename = ('%s_%s%s_%dbit_diff_profiles' % (
            audio_file[:-4], audio_config['signal'].lower(), no_overlapp_text, target_bitwidth)) + coi_suffix
        cv2.imwrite(filename+'.png', diff_profiles_img)
        np.save(filename + '.npy', diff_profiles)
        del diff_profiles_img  # save memory
        del diff_profiles
        # if diff_profiles.shape[1] % 2 == 0:
        #     diff_profiles = np.r_[diff_profiles[:, 1::2], diff_profiles[:, 0::2]]
        # else:
        #     diff_profiles = np.r_[diff_profiles[:, 1::2], diff_profiles[:, 2::2]]
        # diff_profiles = cv2.resize(diff_profiles, (diff_profiles.shape[1] * 2, diff_profiles.shape[0]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Echo profile calculation')
    parser.add_argument(
        '-a', '--audio', help='path to the audio file, .wav or .raw')
    parser.add_argument(
        '-c', '--config', help='path to the config.json file', default='')
    parser.add_argument('-tb', '--target_bitwidth',
                        help='target bitwidth, 2-16', type=int, default=16)
    parser.add_argument(
        '-m', '--maxval', help='maxval for original profiles figure rendering, 0 for adaptive', type=float, default=0)
    parser.add_argument('-md', '--maxdiffval',
                        help='maxval for differential profiles figure rendering, 0 for adaptive', type=float, default=0)
    parser.add_argument('-nd', '--mindiffval',
                        help='maxval for differential profiles figure rendering, 0 for adaptive', type=float, default=0)
    parser.add_argument('-r', '--rectify',
                        help='rectify speaker curve', action='store_true')
    parser.add_argument(
        '--no_overlapp', help='no overlapping while processing frames', action='store_true')
    parser.add_argument(
        '--no_diff', help='do not generate differential echo 1', action='store_true')

    parser.add_argument(
        '--sampling_rate', help='sampling rate of audio file', type=float, default=0)
    parser.add_argument(
        '--n_channels', help='number of channels in audio file', type=int, default=0)
    parser.add_argument('--channels_of_interest',
                        help='channels of interest (starting from 0)', default='')
    parser.add_argument(
        '--frame_length', help='length of each audio frame', type=int, default=0)
    parser.add_argument(
        '--sample_depth', help='sampling depth (bit) of audio file, comma-separated', type=int, default=0)
    parser.add_argument('--bandpass_range',
                        help='bandpass range, LOWCUT,HIGHCUT', default='')

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
    if len(args.channels_of_interest):
        audio_config['channels_of_interest'] = [
            int(x) for x in args.channels_of_interest.split(',')]
    if args.frame_length > 0:
        audio_config['frame_length'] = args.frame_length
    if args.sample_depth > 0:
        audio_config['sample_depth'] = args.sample_depth
    if len(args.bandpass_range):
        audio_config['bandpass_range'] = [
            float(x) for x in args.bandpass_range.split(',')]

    # add suffix to file name to indicate which channels it has. Not used until later, but must be assigned before config is changed.
    channels = ','.join(
        list(map(str, audio_config['channels_of_interest'])))
    coi_suffix = '_' + channels if channels else ''
    echo_profiles(args.audio, audio_config, args.target_bitwidth, args.maxval, args.maxdiffval,
                  args.mindiffval, args.rectify, args.no_overlapp, args.no_diff, coi_suffix)
