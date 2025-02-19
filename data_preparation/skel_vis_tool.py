'''
Visualization tool for selecting the proper view angle for hand skeleton visualization
11/17/2021, Ruidong Zhang, rz379@cornell.edu
'''

import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from visualize_hand import read_recorded_data

def vis_skel(skel, elev, azim):

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

    seq = generate_vis_seq(skel)
    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')

    # adjust the viewing position and angle
    ax.view_init(elev, azim)
    ax.plot(seq[:, 0], seq[:, 1], seq[:, 2], '-o')
    # plt.xlim(seq_ranges_min[0], seq_ranges_max[0])
    # plt.ylim(seq_ranges_min[1], seq_ranges_max[1])
    # ax.set_zlim(seq_ranges_min[2], seq_ranges_max[2])

    plt.show()
    print('%.3f, %.3f' % (ax.elev, ax.azim))
    return ax.elev, ax.azim

def choose_vis_skel(gt_file):

    gt, _ = read_recorded_data(gt_file)
    xmin, xmax = np.min(gt[:, :, 0]), np.max(gt[:, :, 0])
    ymin, ymax = np.min(gt[:, :, 1]), np.max(gt[:, :, 1])
    zmin, zmax = np.min(gt[:, :, 2]), np.max(gt[:, :, 2])

    print(f'({xmin}, {xmax})')
    print(f'({ymin}, {ymax})')
    print(f'({zmin}, {zmax})')
    vis_index = 0
    elev = 20.566
    azim = 98.589
    print('Length:', len(gt))
    while vis_index >= 0:
        this_gt = gt[min(vis_index, len(gt))]
        print('Showing hand pos #', min(vis_index, len(gt)))
        elev, azim = vis_skel(this_gt, elev, azim)
        try:
            vis_index = int(input('Index for vis:'))
        except:
            vis_index = -1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record', help='path to the ground truth pos')

    args = parser.parse_args()

    choose_vis_skel(args.record)