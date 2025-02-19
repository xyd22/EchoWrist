'''
Calculate distance metrics based on predictions
Ruidong Zhang, rz379@cornell.edu, 1/12/2023
'''

import argparse
import numpy as np
import datetime

def distance_metrics(gt_files, pred_files, save_file=''):

    ref_pos = np.array([ 2.92943626e+01, -2.71745727e+01, -1.41075453e+01,  4.70825484e+01,
       -4.90019048e+01, -1.73048689e+01,  6.32941515e+01, -7.56326376e+01,
       -2.63620799e+01,  7.34086326e+01, -9.57408752e+01, -1.91269027e+01,
        2.64356466e+01, -9.16533191e+01, -4.26376199e-01,  3.16875697e+01,
       -1.18127974e+02, -9.22000994e+00,  3.61895103e+01, -1.35876107e+02,
       -1.82267184e+01,  3.98080310e+01, -1.49676501e+02, -4.39218430e+01,
        1.28286073e+00, -9.29130796e+01,  2.43411300e+00,  1.37858225e+00,
       -1.28902727e+02, -8.76364019e+00,  6.86231216e-02, -1.46665725e+02,
       -2.85081149e+01,  1.23909440e+00, -1.65510024e+02, -4.79028986e+01,
       -1.94542727e+01, -8.76374054e+01, -4.39263055e+00, -2.44265846e+01,
       -1.15649655e+02, -1.61715366e+01, -2.43838733e+01, -1.34964383e+02,
       -3.22244634e+01, -2.31955409e+01, -1.53687567e+02, -4.94244405e+01,
       -3.47909021e+01, -7.34604879e+01, -1.08235785e+01, -4.34795598e+01,
       -9.26216878e+01, -1.44591481e+01, -4.96915891e+01, -1.11174418e+02,
       -2.16049949e+01, -5.02390495e+01, -1.26205353e+02, -3.57756265e+01])
    gts = []
    preds = []
    for gt_file, pred_file in zip(gt_files, pred_files):
        gts += [np.load(gt_file)]
        preds += [np.load(pred_file)]
    gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    # preds = np.repeat(ref_pos[None, ...], preds.shape[0], axis=0)

    if gts.shape[1] % 3:    # timestamps are not needed
        ts = gts[:, 0]
        gts = gts[:, 1:]
    else:
        ts = np.zeros(gts.shape[0])
    gts.shape = (gts.shape[0], -1, 3)
    preds.shape = (preds.shape[0], -1, 3)


    # simple vs. complex
    # gts = gts[gts.shape[0]]
    # preds = preds[preds.shape[0]]

    assert(gts.shape[1] == 21 and preds.shape[1] == 20)     # shapes gotta be right

    gt_wrist_points = gts[:, 0:1, :]    # preserve 3 dimensions
    gt_wrist_points = np.repeat(gt_wrist_points, 20, 1)

    gts = gts[:, 1:, :] - gt_wrist_points

    per_point_errors = gts - preds  # N x 20 x 3
    per_point_errors = np.sum(per_point_errors ** 2, axis=2) ** 0.5     # N x 20


    if len(save_file):
        np.savetxt(save_file + '_MJEDE.txt', per_point_errors)

    mean_per_point_error = np.mean(per_point_errors, axis=0)    # 20
    mean_distance_error = np.mean(mean_per_point_error)

    # for e in mean_per_point_error:
    #     print(e, end=' ')
    # print('')

    print('MJEDE: %.2f' % mean_distance_error)

    joint_angle_center = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    joint_angle_previous = [0, 1, 2, 0, 5, 6, 0, 9, 10, 0, 13, 14, 0, 17, 18]
    joint_angle_next = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]

    gt_with_wrist = np.concatenate([np.zeros((gts.shape[0], 1, 3)), gts], axis=1)
    preds_with_wrist = np.concatenate([np.zeros((preds.shape[0], 1, 3)), preds], axis=1)

    gt_joint_angle_vecs_previous = gt_with_wrist[:, joint_angle_previous, :] - gt_with_wrist[:, joint_angle_center, :]
    gt_joint_angle_vecs_next = gt_with_wrist[:, joint_angle_next, :] - gt_with_wrist[:, joint_angle_center, :]
    gt_joint_angles = np.arccos(np.sum(gt_joint_angle_vecs_previous * gt_joint_angle_vecs_next, axis=2) / np.sqrt(np.sum(gt_joint_angle_vecs_previous ** 2, axis=2) * np.sum(gt_joint_angle_vecs_next ** 2, axis=2)))

    preds_joint_angle_vecs_previous = preds_with_wrist[:, joint_angle_previous, :] - preds_with_wrist[:, joint_angle_center, :]
    preds_joint_angle_vecs_next = preds_with_wrist[:, joint_angle_next, :] - preds_with_wrist[:, joint_angle_center, :]
    preds_joint_angles = np.arccos(np.sum(preds_joint_angle_vecs_previous * preds_joint_angle_vecs_next, axis=2) / np.sqrt(np.sum(preds_joint_angle_vecs_previous ** 2, axis=2) * np.sum(preds_joint_angle_vecs_next ** 2, axis=2)))

    joint_angle_errors = np.abs(gt_joint_angles - preds_joint_angles) * 180 / np.pi
    mean_per_joint_angle_error = np.mean(joint_angle_errors, axis=0)
    mean_joint_angle_error = np.mean(mean_per_joint_angle_error)

    if len(save_file):
        np.savetxt(save_file + '_MJAE.txt', joint_angle_errors)

    # for e in mean_per_joint_angle_error:
    #     print(e, end=' ')
    # print('')

    print('MJAE: %.2f' % mean_joint_angle_error)

    file_path = "results.txt"  # Replace with the path to your text file

    # Data to append
    file_name = gt_file.split("/")[3]
    data_to_append = f"{datetime.datetime.now()} {file_name} MJAE: {mean_joint_angle_error:.2f} MJEDE: {mean_distance_error:.2f}"

    # Open the file in append mode and write the data
    with open(file_path, 'a') as file:
        file.write(data_to_append + "\n")  # Add a newline if you want to append a new line

    return per_point_errors, mean_per_point_error, mean_distance_error, joint_angle_errors, mean_per_joint_angle_error, mean_joint_angle_error

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--records', help='paths to the ground truth pos', action='append')
    parser.add_argument('-p', '--preds', help='paths to the predicted pos', action='append')
    parser.add_argument('-o', '--output', help='output error position', default='')

    args = parser.parse_args()

    distance_metrics(args.records, args.preds, args.output)