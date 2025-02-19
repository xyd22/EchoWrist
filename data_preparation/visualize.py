import os
import cv2
import argparse
import numpy as np
import json
import copy

from utils import load_frame_time, display_progress
from data_preparation import load_config
from load_save_gt import load_gt
from echo_profiles import echo_profiles
from visualize_echo_profiles import visualize_echo_profiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Echo profile visualization (with video)')
    parser.add_argument(
        '-p', '--path', help='path to the parent folder of audio files, .wav or .raw')
    # parser.add_argument('-s', '--save', help ='whether or not to save any files that are created in visualization process (e.g. profiles w/ different coi)', default = True)
    parser.add_argument(
        '-c', '--config', help='path to the config.json file', default='')
    parser.add_argument(
        '-r', '--rcds', help='which recordings to visualize, starting from 1, comma-separated, empty = all', default='')
    parser.add_argument('-l', '--echo_length',
                        help='display length of echo profile', type=float, default=8)
    parser.add_argument(
        '--height', help='height of the output video', type=int, default=400)
    parser.add_argument('--pred', help='pred txt file', default='')
    parser.add_argument('-ep', '--echo_profile_path',
                        help='path to the echo_profile image file, default empty for diff', default='')
    parser.add_argument('--poi',
                        help='pixels of interest in one channel, LOWBOUND,HIGHBOUND, default 0,0 for all', default='0,0')
    parser.add_argument('-coi', '--channels_of_interest',
                        help='channels of interest (starting from 0)', default='')
    # all the below arguments are for echo_profiles()
    parser.add_argument('-tb', '--target_bitwidth',
                        help='target bitwidth, 2-16', type=int, default=16)
    parser.add_argument(
        '-m', '--maxval', help='maxval for original profiles figure rendering, 0 for adaptive', type=float, default=0)
    parser.add_argument('-md', '--maxdiffval',
                        help='maxval for differential profiles figure rendering, 0 for adaptive', type=float, default=0)
    parser.add_argument('-nd', '--mindiffval',
                        help='maxval for differential profiles figure rendering, 0 for adaptive', type=float, default=0)
    parser.add_argument('-rect', '--rectify',
                        help='rectify speaker curve', action='store_true')
    parser.add_argument(
        '--no_overlapp', help='no overlapping while processing frames', action='store_true')
    parser.add_argument(
        '--no_diff', help='do not generate differential echo profiles', action='store_true')
    parser.add_argument(
        '--no_video', help='do not generate videos, just create images', action='store_true')

    parser.add_argument(
        '--sampling_rate', help='sampling rate of audio file', type=float, default=0)
    parser.add_argument(
        '--n_channels', help='number of channels in audio file', type=int, default=0)
    parser.add_argument(
        '--frame_length', help='length of each audio frame', type=int, default=0)
    parser.add_argument(
        '--sample_depth', help='sampling depth (bit) of audio file, comma-separated', type=int, default=0)
    parser.add_argument('--bandpass_range',
                        help='bandpass range, LOWCUT,HIGHCUT', default='')
    parser.add_argument(
        '--alternating', help='whether or not to draw alternating profiles', action='store_true')

    args = parser.parse_args()

    config_path = args.config if args.config else os.path.join(
        args.path, 'config.json')
    orig_config = json.load(open(config_path, 'rt'))  # to be restored later
    config = load_config(args.path, args.config)
    audio_config = config['audio']['config']
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
    rcd_oi = [int(x) - 1 for x in rcds.split(',')
              ] if args.rcds else range(len(config['sessions']))
    no_overlapp_text = '_no_overlapp' if args.no_overlapp else ''

    if args.echo_profile_path:
        visualize_echo_profiles(args.path, config_path, args.rcds, args.echo_length, args.height, args.echo_profile_path, tuple(
            [int(x) for x in args.poi.split(',')]), args.pred)
    else:
        try:
            # store temp config so it can be passed into functions
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

            for n_rcd in rcd_oi:
                audio_file_name = config['audio']['files'][n_rcd]
                audio_file = os.path.join(args.path, audio_file_name)
                channels = ','.join(
                    list(map(str, audio_config['channels_of_interest'])))
                coi_suffix = '_' + channels if channels else ''
                echo_profiles(audio_file, audio_config, args.target_bitwidth, args.maxval, args.maxdiffval,
                              args.mindiffval, args.rectify, args.no_overlapp, args.no_diff, coi_suffix, args.alternating)
                # create echo profile, then store in echo_path

            # visualize profile at echo_path
            visualize_echo_profiles(args.path, config_path, args.rcds, args.echo_length, args.height, args.echo_profile_path, tuple(
                [int(x) for x in args.poi.split(',')]), args.pred, coi_suffix)

            # if user does not want to save, delete files created. Not optimally efficient, but otherwise would require refactor of echo_profiles()
            # if not args.save:
            #   abs_path = os.join(args.path, file_name+e)
            #   os.remove(abs_path)
            # restore original config
            with open(config_path, 'w') as f:
                json.dump(orig_config, f, indent=4)

        # always be sure to restore config even in case of errors
        except (KeyboardInterrupt, SystemExit):
            with open(config_path, 'w') as f:
                json.dump(orig_config, f, indent=4)
        except Exception as e:
            with open(config_path, 'w') as f:
                json.dump(orig_config, f, indent=4)
            raise e

    # TODOAFTERPUSH:
    # make sure that if user inputs file with new hyperparameters, it gets recreated
    # no need to call visualize_echo_profiles if the path exists, just show the corresponding video if it exists (the above must be done first)
    # put things into folders
    # delete functionality
