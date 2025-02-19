'''
Rotate a video according to the hand pose in a certain frame
Ruidong Zhang, rz379@cornell.edu, 2/7/2023
'''

import os
import cv2
import argparse
import numpy as np
import mediapipe as mp


def rotate_video(video_path, anchor_frame):

    renamed_file = video_path[:-4] + '_original' + video_path[-4:]
    os.rename(video_path, renamed_file)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(renamed_file)

    cap.set(cv2.CAP_PROP_POS_FRAMES, anchor_frame - 1)

    _, frame = cap.read()

    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5) as hands:
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        image_height, image_width, _ = frame.shape

        hand_landmarks = results.multi_hand_landmarks[0]

        middle_finger_tip_pos = (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)
        wrist_pos = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)

        rotation_center = ((middle_finger_tip_pos[0] + wrist_pos[0]) / 2, (middle_finger_tip_pos[1] + wrist_pos[1]) / 2)
        cv2.circle(frame, tuple([int(x) for x in middle_finger_tip_pos]), 2, (255, 0, 0))
        cv2.circle(frame, tuple([int(x) for x in wrist_pos]), 2, (0, 255, 0))

        dir_vec = np.array([middle_finger_tip_pos[0] - wrist_pos[0], middle_finger_tip_pos[1] - wrist_pos[1]])
        dir_vec /= np.math.sqrt(np.sum(dir_vec ** 2))
        rotation_angle = np.math.acos(-dir_vec[1]) * 180 / np.pi

        rot_mat = cv2.getRotationMatrix2D(rotation_center, -rotation_angle, 1.0)
        rotated_frame = cv2.warpAffine(frame, rot_mat, frame.shape[1::-1], flags=cv2.INTER_CUBIC)
        # cv2.imwrite('marked.png', rotated_frame)

        vid = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), 30, frame.shape[1::-1])

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        grabbed, frame = cap.read()
        while grabbed:
            rotated_frame = cv2.warpAffine(frame, rot_mat, frame.shape[1::-1], flags=cv2.INTER_CUBIC)
            vid.write(rotated_frame)
            grabbed, frame = cap.read()
        
        cap.release()
        vid.release()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', help='path to the video file')
    parser.add_argument('-f', '--anchor-frame', type=int, help='number of frame used as the anchor')

    args = parser.parse_args()
    rotate_video(args.video, args.anchor_frame)

