'''
Visualizing the syncing the hand (ground truth and/or pred) with the recording video
8/19/2021
Ruidong Zhang, rz379@cornell.edu
'''

import os
import cv2
import argparse
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import cpu_count
from multiprocessing import Pool as ProcessPool
import matplotlib

from utils import load_frame_time
# import matplotlib.animation as animation

matplotlib.use("Agg") # weird thing about fig.canvas.tostring_rgb(), https://stackoverflow.com/questions/67601709/fig-canvas-tostring-rgb-output-wrong-length-bytes-but-only-on-windows

def draw_frame_and_write(item):

    def generate_vis_seq(d):
        vis_idx = [
            0,
            1, 2, 3, 4, 3, 2, 1,
            5, 6, 7, 8, 7, 6, 5,
            9, 10, 11, 12, 11, 10, 9,
            13, 14, 15, 16, 15, 14, 13,
            17, 18, 19, 20, 19, 18, 17,
            0
        ]
        return d[vis_idx]

    def get_dummy_hand(gt_seq, pred_seq, foi):
        if len(foi) == 0:
            return pred_seq
        def get_poi(finger_idx):
            return list(range((finger_idx - 1) * 4 + 1, finger_idx * 4 + 1))
        poi = []        # list of point index
        for f in foi:
            poi += get_poi(f)
        assert(len(pred_seq) == len(poi))   # points must match
        dummy_hand = deepcopy(gt_seq)
        for n, i in enumerate(poi):
            dummy_hand[i] = deepcopy(pred_seq[n])
        return dummy_hand
    original_seq, original_seq_pred, foi, seq_ranges_min, seq_ranges_max, idx, save_pos, views = item
    seq = generate_vis_seq(original_seq)
    seq_plt = seq
    seq_pred = None
    fig = plt.figure(idx, figsize=(6, 6 * len(views)))
    fig.clf()
    if original_seq_pred is not None:
        seq_pred = generate_vis_seq(get_dummy_hand(original_seq, original_seq_pred, foi))
    else:
        seq_pred = seq
    #     ax = fig.add_subplot(len(views), 1, 1, projection='3d')
    # else:

    # adjust the viewing position and angle
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(len(views), 1, i + 1, projection='3d')
        ax.view_init(elev, azim)
        ax.plot(seq_pred[:, 0], seq_pred[:, 1], seq_pred[:, 2], '-o', color='brown')
        ax.plot(seq_plt[:, 0], seq_plt[:, 1], seq_plt[:, 2], '-o',color='green')
        plt.xlim(seq_ranges_min[0], seq_ranges_max[0])
        plt.ylim(seq_ranges_min[1], seq_ranges_max[1])
        ax.set_zlim(seq_ranges_min[2], seq_ranges_max[2])
    # if seq_pred is not None:
    #     ax = fig.add_subplot(212, projection='3d')
    #     plt.plot(seq_pred[:, 0], seq_pred[:, 1], seq_pred[:, 2], '-ro')
    #     elev = 20.566
    #     azim = 98.589
    #     ax.view_init(elev, azim)
    #     plt.xlim(seq_ranges_min[0], seq_ranges_max[0])
    #     plt.ylim(seq_ranges_min[1], seq_ranges_max[1])
    #     ax.set_zlim(seq_ranges_min[2], seq_ranges_max[2])
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    # fig.show()
    # fig.savefig('%s/%06d.png' % (save_pos, idx), bbox_inches = 'tight', pad_inches = 0)
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return img


def generate_frames(drawing_sequences, save_pos, drawing_sequences_pred=None, gt_video='', gt_video_matched_frames=[], foi=[]):
    if not os.path.exists(save_pos):
        print('Creating', save_pos)
        os.mkdir(save_pos)
    print(drawing_sequences.shape)
    def min_max_range(seq):
        mins = np.array([np.min(seq[:, :, i]) for i in range(3)])
        maxs = np.array([np.max(seq[:, :, i]) for i in range(3)])
        half_ranges = (maxs - mins) * 1.1 / 2
        mids = (mins + maxs) / 2
        ranges_min = mids - half_ranges
        ranges_max = mids + half_ranges
        return ranges_min, ranges_max
    seq_ranges_min, seq_ranges_max = min_max_range(drawing_sequences)
    if drawing_sequences_pred is not None:
        pred_ranges_min, pred_ranges_max = min_max_range(drawing_sequences_pred)
        seq_ranges_min = np.min([seq_ranges_min, pred_ranges_min], axis=0)
        seq_ranges_max = np.max([seq_ranges_max, pred_ranges_max], axis=0)

    views = [
        (-80, -85.843),
        # (-118.233, -85.843),
        # (28.212103896104963, -93.53762337662488),
    ]

    items = [(drawing_sequences[i], drawing_sequences_pred[i] if (drawing_sequences_pred is not None and i < len(drawing_sequences_pred)) else None, foi, seq_ranges_min, seq_ranges_max, i, save_pos, views) for i in range(len(drawing_sequences))]

    n_cpus = int(cpu_count() * 0.75)     # 75% of all cpus
    print('Using %d cpus' % n_cpus)
    pool = ProcessPool(n_cpus)
    results = pool.map(draw_frame_and_write, items)
    pred_target_size = results[0].shape[:2][::-1]

    cap = cv2.VideoCapture(gt_video)
    cap_original_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap_target_size = (int(cap_original_size[0] * pred_target_size[1] / cap_original_size[1]), pred_target_size[1])
    target_size = (cap_target_size[0] + pred_target_size[0], pred_target_size[1])

    # print(pred_target_size, cap_original_size, cap_target_size, target_size)

    cap_n_frame = 0
    pred_n_frame = 0

    vid = cv2.VideoWriter(save_pos + '.mp4', cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), 30, target_size)

    success, frame = cap.read()
    while success:
        if cap_n_frame in gt_video_matched_frames:
            video_feed = cv2.resize(frame, cap_target_size)
            # print(video_feed.shape)
            assembled_frame = np.concatenate([video_feed[:,::-1], results[pred_n_frame]], axis=1)
            vid.write(assembled_frame)
            pred_n_frame += 1
        success, frame = cap.read()
        cap_n_frame += 1

    cap.release()
    vid.release()

def read_recorded_data(save_path):
    raw = np.load(save_path)
    if raw.shape[1] % 3 != 0:
        all_data = raw[:, 1:]
        ts = raw[:, 0]
    else:
        all_data = raw
        ts = None
    all_seq = []
    # no palm point, add zero as the palm point
    if all_data.shape[1] == 3 * 20:
        all_data = np.c_[np.zeros((all_data.shape[0], 3)), all_data]
    for frame_data in all_data:
        reshaped_frame_data = np.reshape(frame_data, (frame_data.shape[0] // 3, 3))
        reshaped_frame_data -= reshaped_frame_data[0]
        # d = reshaped_frame_data
        
        all_seq += [reshaped_frame_data]
    return np.array(all_seq), ts


def match_frames(ts_video, ts_pred):
    ts_pred_idx = np.arange(0, ts_pred.shape[0])
    ts_pred_sorted = np.array([ts_pred, ts_pred_idx]).T
    ts_pred_sorted = ts_pred_sorted[ts_pred_sorted[:, 0].argsort()]

    vid_idx = 0
    pred_idx = 0
    vid_matched_idx = []
    pred_match_idx = []

    eps = 25e-3         # within 25ms consider the same frame
    while pred_idx < ts_pred.shape[0] and vid_idx < len(ts_video):
        if abs(ts_video[vid_idx] - ts_pred_sorted[pred_idx, 0]) < eps:
            vid_matched_idx += [vid_idx]
            pred_match_idx += [int(ts_pred_sorted[pred_idx, 1])]
            vid_idx += 1
            pred_idx += 1
        elif ts_pred_sorted[pred_idx, 0] < ts_video[vid_idx]:
            while pred_idx < ts_pred.shape[0] and ts_pred_sorted[pred_idx, 0] < ts_video[vid_idx] - eps:
                pred_idx += 1
        else:
            while vid_idx < len(ts_video) and ts_video[vid_idx] < ts_pred_sorted[pred_idx, 0] - eps:
                vid_idx += 1

    return vid_matched_idx, pred_match_idx


def visualize_hand(record_pos, output_pos, pred_pos='', gt_video='', foi=''):
    seq, ts = read_recorded_data(record_pos)
    # np.savetxt(os.path.abspath(output_pos) + '_frame_time.txt', ts, delimiter=' ')
    pred_seq = None
    if len(pred_pos):
        pred_seq, _ = read_recorded_data(pred_pos)

    if len(gt_video):
        assert(ts is not None)  # timestamp is needed to generate a side-by-side video
        gt_video_ts = load_frame_time(gt_video[:-4] + '_frame_time.txt')
        vid_matched_idx, pred_match_idx = match_frames(gt_video_ts, ts)
        # print(vid_matched_idx, pred_match_idx)
        seq = seq[pred_match_idx]
        ts = ts[pred_match_idx]
        if pred_seq is not None:
            pred_seq = pred_seq[pred_match_idx]
    generate_frames(seq, output_pos, pred_seq, gt_video, vid_matched_idx, [int(x) for x in foi.split(',')] if len(foi) else [])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record', help='path to the ground truth pos')
    parser.add_argument('-p', '--pred', help='path to the predicted pos', default='')
    parser.add_argument('-v', '--video', help='path to the ground truth video to be placed beside predictions', default='')
    parser.add_argument('-o', '--output', help='output folder position')
    parser.add_argument('-f', '--foi', help='finger of interest in pred data', default='')
    # args.add_argument('-o', '--output', default='temp', help='minimum area size')

    args = parser.parse_args()

    visualize_hand(args.record, args.output, args.pred, args.video, args.foi)