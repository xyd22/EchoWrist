'''
Identify the facial landmarks from a video and save it to .npy file
2/27/2022, Ruidong Zhang, rz379@cornell.edu
'''

import cv2
import dlib
import argparse
import numpy as np
from copy import deepcopy
from utils import display_progress
from multiprocessing import cpu_count
from multiprocessing import Pool as ProcessPool

from utils import load_frame_time

def calibration(q1, q2, q3, p1, p2, p3):
    a = []
    b = []

    p = np.array([[p1[0], p1[1], 1], [p2[0], p2[1], 1], [p3[0], p3[1], 1]])
    qx = np.array([[q1[0]], [q2[0]], [q3[0]]])
    qy = np.array([[q1[1]], [q2[1]], [q3[1]]])

    try:
        p_inv = np.linalg.inv(p)
        a = np.matmul(p_inv, qx)
        b = np.matmul(p_inv, qy)
        #print(a, b)
        return a, b
    except np.linalg.LinAlgError as e:
        print(e)
        return False


# convert the coordinate of camera to the coordinate of projection
def convert_position(position, a, b):
    """
    param: position shape: n*2
    return: points shape: n*2
    """
    p = np.array([float(position[0]), float(position[1]), 1.])
    points = [np.matmul(p, a), np.matmul(p, b)]
    return points


def convert_mouth(points):
    converted = deepcopy(points)
    zero = points[33]
    achor1 = points[39]
    achor2 = points[42]
    q1, q2, q3 = ([317, 305], [278, 228], [353, 228])   #Config.achor
    a, b = calibration(q1, q2, q3, zero, achor1, achor2)
    for i in range(0, 68):
        converted[i] = convert_position(points[i], a, b)
    points_of_interest = list(range(48, 68)) + list(range(3, 14))           # mouth and lower face

    mouth_center = np.mean(converted[48:68], axis=0)
    converted -= mouth_center
    return converted[points_of_interest]

def label_img(item):
    img, detector, predictor = item
    dets = detector(img, 1)
    points = np.zeros((68, 2))
    dets_area = [(dets[i].width() * dets[i].height(), i, dets[i]) for i in range(len(dets))]
    if len(dets_area):
        shape = predictor(img, max(dets_area)[2])        # only label the largets detection
        for i in range(0, 68):
            points[i, 0] = shape.part(i).x
            points[i, 1] = shape.part(i).y
        mouth_points = convert_mouth(points)
        return points, mouth_points
    return None, None

def facial_landmarks(video_path, buffer_length=5000):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    frame_times = load_frame_time(video_path[:-4] + '_frame_time.txt')

    cap = cv2.VideoCapture(video_path)
    n_frames = 0
    gt = []
    gt_poi = []

    n_cpus = int(cpu_count() * 0.75)     # 75% of all cpus
    print('Using %d cpus' % n_cpus)
    pool = ProcessPool(n_cpus)
    buffer_frames = None
    buffer_pos = 0
    while True:
        success, frame = cap.read()
        # if not success:
        #     break
        if success:
            if buffer_frames is None:
                buffer_frames = np.zeros((buffer_length, frame.shape[0], frame.shape[1], frame.shape[2]), frame.dtype)
            buffer_frames[buffer_pos] = frame
            buffer_pos += 1
        if buffer_pos == buffer_length or not success:
            results = pool.map(label_img, [(buffer_frames[x], detector, predictor) for x in range(buffer_pos)])
            buffer_pos = 0
            # display_progress(n_frames, len(frame_times), 50)
            # points, points_of_interests = label_img(frame, detector, predictor)
            for points, points_of_interests in results:
                display_progress(n_frames, len(frame_times), 500)
                if points_of_interests is not None:
                    gt += [[frame_times[n_frames]] + list(points.ravel())]
                    gt_poi += [[frame_times[n_frames]] + list(points_of_interests.ravel())]
                n_frames += 1
            if not success:
                break

    gt = np.array(gt)
    gt_poi = np.array(gt_poi)
    np.save(video_path[:-4] + '_landmarks.npy', gt)
    np.save(video_path[:-4] + '_poi_landmarks.npy', gt_poi)
    return gt, gt_poi

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', help='path to the video file')

    args = parser.parse_args()
    facial_landmarks(args.video)
