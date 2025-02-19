'''
Calculate distance metrics based on predictions
Ruidong Zhang, rz379@cornell.edu, 1/12/2023
'''

import argparse
import numpy as np

def wrist_rot_metric(gt_files, pred_files, save_file=''):
    gts = []
    preds = []
    for gt_file, pred_file in zip(gt_files, pred_files):
        gts += [np.load(gt_file)]
        preds += [np.load(pred_file)]
    gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    if gts.shape[1] % 3:    # timestamps are not needed
        ts = gts[:, 0]
        gts = gts[:, 1:]
    else:
        ts = np.zeros(gts.shape[0])
    gts.shape = (gts.shape[0], -1, 3)
    preds.shape = (preds.shape[0], -1, 3)

    assert(gts.shape[1] == 21 and preds.shape[1] == 20)     # TOCHANGE: shapes gotta be right 1 / 21

    gts = gts[:, 9, :]
    preds = preds[:, 8, :]    # TOCHANGE: 0 / 9

    angular_error = np.arccos(np.sum(gts * preds, axis=1) / np.sqrt(np.sum(gts ** 2, axis=1) * np.sum(preds ** 2, axis=1)))
    angular_error = angular_error * 180 / np.pi

    if len(save_file):
        np.savetxt(save_file, angular_error)

    mean_wrist_angular_error = np.mean(angular_error, axis=0)    # 20

    print('MWAE: %.2f' % mean_wrist_angular_error)

    return angular_error, mean_wrist_angular_error


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--records', help='paths to the ground truth pos', action='append')
    parser.add_argument('-p', '--preds', help='paths to the predicted pos', action='append')
    parser.add_argument('-o', '--output', help='output error position', default='')

    args = parser.parse_args()

    wrist_rot_metric(args.records, args.preds, args.output)