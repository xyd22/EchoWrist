'''
Pre-process data and generate dataset for training
2/17/2022, Ruidong Zhang, rz379@cornell.edu
'''

import os
import json
import warnings
import argparse
import numpy as np

from utils import load_frame_time, load_config
from load_save_gt import load_gt, save_gt
from echo_profiles import echo_profiles
from mp_ground_truth import mp_ground_truth
from facial_landmarks import facial_landmarks
from corrupted_sample_parser import parse_corrupted_record

from normalize_video import normalize_landmarks

def ts_to_idx(ts, all_ts):
    return np.argmin(np.abs(all_ts - ts))

def data_preparation(parent_folder, config_file='', force_overwrite=False, response_offset=0.2, target_bitwidth=16, maxval=None, maxdiff=None, mindiff=None, rectify=False, no_overlapp=False, no_diff=False):
    
    config = load_config(parent_folder, config_file)
    # deal with audios first
    audio_config = config['audio']['config']
    for f in config['audio']['files']:
        audio_path = os.path.join(parent_folder, f)
        if not os.path.exists(audio_path):
            warnings.warn('Warning: audio file %s is specified in config file but was not found, skipped' % f, RuntimeWarning)
            continue
        if (not os.path.exists('%s_%s_%dbit_profiles.npy' % (audio_path[:-4], audio_config['signal'].lower(), target_bitwidth))) or force_overwrite:
            echo_profiles(audio_path, audio_config, target_bitwidth, maxval, maxdiff, mindiff, rectify, no_overlapp, no_diff)

    # generate landmarks based on video
    if 'facial_landmarks' in config['tasks']:
        for f in config['ground_truth']['videos']:
            video_path = os.path.join(parent_folder, f)
            if not os.path.exists(video_path):
                warnings.warn('Warning: video file %s is specified in config file but was not found, skipped' % f, RuntimeWarning)
                continue
            if not os.path.exists('%s_facial_landmarks.npy' % (video_path[:-4])):
                print('Generating facial landmarks for %s' % video_path)
                facial_landmarks(video_path)

    # generate hand landmarks based on video
    if 'hand_landmarks' in config['tasks']:
        for f in config['ground_truth']['videos']:
            video_path = os.path.join(parent_folder, f)
            if not os.path.exists(video_path):
                warnings.warn('Warning: video file %s is specified in config file but was not found, skipped' % f, RuntimeWarning)
                continue
            if not os.path.exists('%s_landmarks.npy' % (video_path[:-4])):
                print('Generating hand landmarks for %s' % video_path)
                mp_ground_truth(video_path)
            if 'anchor_length' in config['ground_truth']:
                normalize_landmarks('%s_landmarks.npy' % (video_path[:-4]), 'anchor_frame.png', config['ground_truth']['anchor_length'])

    # find all the _profiles.npy files
    all_profile_npys = [x for x in os.listdir(parent_folder) if x[-13:] == '_profiles.npy']

    # if there are corrupted samples, identify and remove them.
    corrupted_ts_ranges = parse_corrupted_record(parent_folder, config_file)

    for n_rcd, rcd_sessions in enumerate(config['sessions']):
        # syncing poses in the format of frames
        gt_syncing_pos = config['ground_truth']['syncing_poses'][n_rcd]
        if abs(gt_syncing_pos - round(gt_syncing_pos)) < 1e-6:
            gt_ts_file = os.path.join(parent_folder, config['ground_truth']['videos'][n_rcd][:-4] + '_frame_time.txt')
            gt_video_ts = load_frame_time(gt_ts_file)
            gt_syncing_pos = gt_video_ts[gt_syncing_pos - 1]
            video_start_ts = gt_video_ts[0]
        else:
            video_start_ts = gt_syncing_pos
        audio_syncing_pos = config['audio']['syncing_poses'][n_rcd]
        if audio_syncing_pos < config['audio']['config']['frame_length'] * 50:
            audio_syncing_pos *= config['audio']['config']['frame_length']
        #if frame_ratio > 1:
        #    audio_syncing_pos = (audio_syncing_pos // frame_ratio) if is_stack else (audio_syncing_pos * frame_ratio)
        
        rcd_config = {
            'audio_config': audio_config,
            'syncing': {
                'audio': audio_syncing_pos,
                'ground_truth': gt_syncing_pos,
            },
            'response_offset': response_offset
        }

        gts = []
        if 'classification' in config['tasks']:
            gt_file = os.path.join(parent_folder, config['ground_truth']['files'][n_rcd])
            gt_classification = load_gt(gt_file)
            gt_start_ts = np.array([float(x[1]) for x in gt_classification])
            gt_end_ts = np.array([float(x[2]) for x in gt_classification])
            gts += [{
                'gt': gt_classification,
                'target': 'ground_truth_classification.txt',
                'start_ts': gt_start_ts,
                'end_ts': gt_end_ts
            }]
        if 'blendshapes' in config['tasks']:
            gt_file = os.path.join(parent_folder, config['ground_truth']['files'][n_rcd])
            gt_blendshape = load_gt(gt_file)
            gt_ts = np.array([float(x[0]) for x in gt_blendshape])
            gts += [{
                'gt': gt_blendshape,
                'target': 'ground_truth_blendshapes.npy',
                'start_ts': gt_ts,
                'end_ts': gt_ts
            }]
        if 'facial_landmarks' in config['tasks'] or 'hand_landmarks' in config['tasks']:
            gt_file = os.path.join(parent_folder, config['ground_truth']['videos'][n_rcd])[:-4] + '_landmarks.npy'
            gt_landmarks = load_gt(gt_file)
            gt_ts = np.array([x[0] for x in gt_landmarks])
            gts += [{
                'gt': gt_landmarks,
                'target': 'ground_truth_landmarks.npy',
                'start_ts': gt_ts,
                'end_ts': gt_ts
            }]
            gt_norm_file = os.path.join(parent_folder, config['ground_truth']['videos'][n_rcd])[:-4] + '_landmarks_normalized.npy'
            if os.path.exists(gt_norm_file):
                gt_norm = load_gt(gt_norm_file)
                gts += [{
                    'gt': gt_norm,
                    'target': 'ground_truth_poi_landmarks_normalized.npy',
                    'start_ts': gt_ts,
                    'end_ts': gt_ts
                }]
        if 'regression' in config['tasks']:
            gt_file = os.path.join(parent_folder, config['ground_truth']['files'][n_rcd])
            gt_regression = load_gt(gt_file)
            gt_ts = np.array([x[0] for x in gt_regression])
            gts += [{
                'gt': gt_regression,
                'target': 'ground_truth_regression.npy',
                'start_ts': gt_ts,
                'end_ts': gt_ts
            }]

        for n_ss, ss in enumerate(rcd_sessions):
            print('Dealing with session %02d in recording %02d, ' % (n_ss + 1, n_rcd + 1), end='')
            session_target = os.path.join(parent_folder, 'dataset', 'session_%02d%02d' % (n_rcd + 1, n_ss + 1))
            if not os.path.exists(session_target):
                os.makedirs(session_target)
            if ss['start'] < 1600000000:
                ss_s = video_start_ts + ss['start']
            else:
                ss_s = ss['start']
            ss_e = ss_s + ss['duration']

            # writing and linking files
            config_target = os.path.join(session_target, 'config.json')
            print('    Writing session config at %s' % config_target)
            json.dump(rcd_config, open(config_target, 'wt'), indent=4)
            for npy in all_profile_npys:
                if npy[:len(config['audio']['files'][n_rcd]) - 4] == config['audio']['files'][n_rcd][:-4]:
                    target_npy = os.path.join(session_target, npy[len(config['audio']['files'][n_rcd]) - 3:])
                    if not os.path.exists(target_npy):
                        print('    Linking %s -> %s' % (npy, target_npy))
                        os.symlink(os.path.join('../../', npy), target_npy)
            
            for this_gt in gts:
                ss_s_idx = ts_to_idx(ss_s, this_gt['start_ts'])
                ss_e_idx = ts_to_idx(ss_e, this_gt['end_ts'])
                session_gt = []
                for i in range(ss_s_idx, ss_e_idx + 1):
                    item_not_corrupted = True
                    for corrupted_s, corrupted_e in corrupted_ts_ranges:
                        if max(corrupted_s, this_gt['start_ts'][i]) <= min(corrupted_e, this_gt['end_ts'][i]):
                            item_not_corrupted = False
                            break
                    if item_not_corrupted:
                        session_gt += [this_gt['gt'][i]]
                # print('ground truth length: %d' % (len(session_gt)))

                gt_target = os.path.join(session_target, this_gt['target'])
                print('    Writing ground truth at %s, length %d' % (gt_target, len(session_gt)))
                save_gt(session_gt, gt_target)

            ref_video_source_actual = os.path.join(parent_folder, config['ground_truth']['videos'][n_rcd])
            ref_video_source = os.path.join('../../', config['ground_truth']['videos'][n_rcd])
            ref_video_target = os.path.join(session_target, 'reference_video.mp4')
            if os.path.exists(ref_video_source_actual) and (not os.path.exists(ref_video_target)):
                print('    Linking %s -> %s' % (config['ground_truth']['videos'][n_rcd], ref_video_target))
                os.symlink(ref_video_source, ref_video_target)
                os.symlink(ref_video_source[:-4] + '_frame_time.txt', ref_video_target[:-4] + '_frame_time.txt')
            
            facial_landmarks_source_actual = os.path.join(parent_folder, config['ground_truth']['videos'][n_rcd][:-4] + '_facial_landmarks.npy')
            facial_landmarks_source = os.path.join('../../', config['ground_truth']['videos'][n_rcd][:-4] + '_facial_landmarks.npy')
            facial_landmarks_target = os.path.join(session_target, 'facial_landmarks.npy')
            if os.path.exists(facial_landmarks_source_actual) and (not os.path.exists(facial_landmarks_target)):
                print('    Linking %s -> %s' % (config['ground_truth']['videos'][n_rcd][:-4] + '_facial_landmarks.npy', facial_landmarks_target))
                os.symlink(facial_landmarks_source, facial_landmarks_target)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Pre-process data and generate dataset for training')
    parser.add_argument('-p', '--path', help='path to the folder where files are saved (and dataset will be saved)')
    parser.add_argument('-c', '--config', help='path to the config.json file', default='')
    parser.add_argument('-f', '--force', help='force overwrite echo profile files', action='store_true')
    parser.add_argument('--response_offset', help='response time offset (s)', type=float, default=0.2)
    parser.add_argument('-tb', '--target_bitwidth', help='target bitwidth, 2-16', type=int, default=16)
    parser.add_argument('-m', '--maxval', help='maxval for original profiles figure rendering, 0 for adaptive', type=int, default=0)
    parser.add_argument('-md', '--maxdiffval', help='maxval for differential profiles figure rendering, 0 for adaptive', type=int, default=0)
    parser.add_argument('-nd', '--mindiffval', help='maxval for differential profiles figure rendering, 0 for adaptive', type=int, default=0)
    parser.add_argument('-r', '--rectify', help='rectify speaker curve', action='store_true')
    parser.add_argument('--no_overlapp', help='no overlapping while processing frames', action='store_true')
    parser.add_argument('--no_diff', help='do not generate differential echo profiles', action='store_true')
    # parser.add_argument('--stack', help='stack multiple frames or split frames', action='store_true')
    # parser.add_argument('--fr', help='frame_ratio', type=int, default=1)

    args = parser.parse_args()
    data_preparation(args.path, args.config, args.force, args.response_offset, args.target_bitwidth, args.maxval, args.maxdiffval, args.mindiffval, args.rectify, args.no_overlapp, args.no_diff)