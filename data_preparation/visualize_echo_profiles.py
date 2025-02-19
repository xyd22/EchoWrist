'''
Visualize echo profiles together with ground-truth video
2/27/2022, Ruidong Zhang, rz379@cornell.edu
'''

import os
import cv2
import argparse
import numpy as np

from utils import load_frame_time, display_progress
from data_preparation import load_config
from load_save_gt import load_gt


def visualize_echo_profiles(parent_folder, config_file='', rcds='', echo_display_length=5, output_height=400, echo_profile_path='', poi=(0, 0), preds='', coi_suffix='', alternating=False):
    config = load_config(parent_folder, config_file)
    audio_config = config['audio']['config']
    if len(rcds):   # note: rcds start from 1, aligned with session split
        rcd_oi = [int(x) - 1 for x in rcds.split(',')]
    else:
        rcd_oi = range(len(config['sessions']))
    if len(echo_profile_path):
        # obviously one echo_profile can only match one video
        assert (len(rcd_oi) == 1)
    if len(preds):
        loaded_preds = load_gt(preds)
        pred_s_time = [float(x[1]) for x in loaded_preds]
        pred_e_time = [float(x[2]) for x in loaded_preds]
        pred_cur_idx = 0
    for n_rcd in range(len(config['sessions'])):
        if n_rcd not in rcd_oi:
            continue
        gt_ts_file = os.path.join(
            parent_folder, config['ground_truth']['videos'][n_rcd][:-4] + '_frame_time.txt')
        video_ts = load_frame_time(gt_ts_file)
        # syncing poses in the format of frames
        gt_syncing_pos = config['ground_truth']['syncing_poses'][n_rcd]
        if gt_syncing_pos < 1600000000:
            gt_syncing_pos = video_ts[gt_syncing_pos - 1]
        audio_syncing_frame = config['audio']['syncing_poses'][n_rcd]
        if audio_syncing_frame >= audio_config['frame_length'] * 50:
            audio_syncing_frame = audio_syncing_frame // audio_config['frame_length']

        if len(echo_profile_path) == 0:
            if not alternating:
                rcd_echo_profile_path = '%s/%s_%s_16bit_diff_profiles%s.png' % (
                    parent_folder, config['audio']['files'][n_rcd][:-4], audio_config['signal'].lower(), coi_suffix)
            else:
                rcd_echo_profile_path = '%s/%s_%s_16bit_alternating_profiles%s.png' % (
                    parent_folder, config['audio']['files'][n_rcd][:-4], audio_config['signal'].lower(), coi_suffix)

        else:
            rcd_echo_profile_path = echo_profile_path
        echo_profile_img = cv2.imread(rcd_echo_profile_path)

        n_channels = len(audio_config['channels_of_interest'])
        # data size must match the channels
        assert (echo_profile_img.shape[0] % n_channels == 0)
        profile_channel_height = echo_profile_img.shape[0] // n_channels
        if poi != (0, 0):
            rows_of_interest = []
            for i in range(n_channels):
                rows_of_interest += list(range(profile_channel_height *
                                         i + poi[0], profile_channel_height * i + poi[1]))
            echo_profile_img = echo_profile_img[rows_of_interest, :]

        echo_display_frames = round(
            echo_display_length * audio_config['sampling_rate'] / audio_config['frame_length'])
        echo_profile_img = np.concatenate([np.zeros((echo_profile_img.shape[0], echo_display_frames,
                                          echo_profile_img.shape[2]), dtype=echo_profile_img.dtype), echo_profile_img], axis=1)
        echo_profile_reshape_target_size = (round(
            echo_display_frames * output_height / echo_profile_img.shape[0]), output_height)

        cap = cv2.VideoCapture(os.path.join(
            parent_folder, config['ground_truth']['videos'][n_rcd]))
        n_frames = 0
        success, frame = cap.read()
        source_img_reshape_target_size = (
            round(frame.shape[1] * output_height / frame.shape[0]), output_height)
        target_size = (
            echo_profile_reshape_target_size[0] + source_img_reshape_target_size[0], output_height)
        taret_path = '%s/%s_%s.mp4' % (parent_folder, config['ground_truth']
                                       ['videos'][n_rcd][:-4], os.path.basename(rcd_echo_profile_path)[:-4])
        print('Writing to %s, size %d x %d' %
              (os.path.abspath(taret_path), target_size[0], target_size[1]))
        vid = cv2.VideoWriter(taret_path, cv2.VideoWriter_fourcc(
            'a', 'v', 'c', '1'), 30, target_size)
        while success and n_frames < len(video_ts):
            display_progress(n_frames, len(video_ts), 1000)
            frame_text1 = ''
            frame_text2 = ''
            if len(preds):
                if pred_e_time[pred_cur_idx] < video_ts[n_frames]:
                    pred_cur_idx += 1
                if pred_cur_idx < len(pred_s_time) and pred_s_time[pred_cur_idx] <= video_ts[n_frames] <= pred_e_time[pred_cur_idx]:
                    frame_text1 = 't: ' + loaded_preds[pred_cur_idx][0]
                    frame_text2 = 'p: ' + loaded_preds[pred_cur_idx][-1]
            if len(frame_text1):
                cv2.putText(frame, frame_text1, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)
                cv2.putText(frame, frame_text2, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)
            echo_profile_pos_of_frame = round(
                (video_ts[n_frames] - gt_syncing_pos) * audio_config['sampling_rate'] / audio_config['frame_length'] + audio_syncing_frame)
            n_frames += 1
            if echo_profile_pos_of_frame < 0 or echo_profile_pos_of_frame + echo_display_frames > echo_profile_img.shape[1]:
                success, frame = cap.read()
                continue
            echo_profile_frame = echo_profile_img[:,
                                                  echo_profile_pos_of_frame: echo_profile_pos_of_frame + echo_display_frames, :]
            target_frame = np.concatenate([cv2.resize(echo_profile_frame, echo_profile_reshape_target_size), cv2.resize(
                frame, source_img_reshape_target_size)], axis=1)
            vid.write(target_frame)
            success, frame = cap.read()
        cap.release()
        vid.release()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Echo profile visualization (with video)')
    parser.add_argument(
        '-p', '--path', help='path to the audio file, .wav or .raw')
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
    parser.add_argument(
        '--poi', help='pixels of interest in one channel, LOWBOUND,HIGHBOUND, default 0,0 for all', default='0,0')

    args = parser.parse_args()

    visualize_echo_profiles(args.path, args.config, args.rcds, args.echo_length, args.height,
                            args.echo_profile_path, tuple([int(x) for x in args.poi.split(',')]), args.pred)
