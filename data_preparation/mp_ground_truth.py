'''
Data labeling using MediaPipe
2/21/2022, Ruidong Zhang, rz379@cornell.edu
ref: https://google.github.io/mediapipe/solutions/hands#python-solution-api
'''

import cv2
import argparse
import numpy as np
import mediapipe as mp

from utils import display_progress

def mp_ground_truth(video_path):

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(video_path)
    n_frames = 0
    ts = []
    with open(video_path[:-4] + '_frame_time.txt', 'rt') as f:
        for l in f.readlines():
            if l[0] < '0' or l[0] > '9':
                continue
            ts += [float(l)]
    save_data = []
    vid = cv2.VideoWriter(video_path[:-4] + '_gt_marked.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))

    try:
    # if True:
        with mp_hands.Hands(
                model_complexity=1,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while True:
                display_progress(n_frames, len(ts))
                success, image = cap.read()
                if not success:
                    # print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frame_ts = ts[n_frames]
                n_frames += 1
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                frame_landmarks = None
                if results.multi_hand_landmarks:
                    frame_landmarks = np.zeros((21, 3))
                    for hand_landmarks in results.multi_hand_world_landmarks:
                        for i, lm in enumerate(hand_landmarks.landmark):
                            frame_landmarks[i] = lm.x, lm.y, lm.z
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                    frame_landmarks *= 1000      # meter to millimeters
                if frame_landmarks is not None:
                    frame_data = np.zeros((64,))
                    frame_data[0] = frame_ts
                    frame_data[1:] = frame_landmarks.ravel()
                    save_data += [frame_data]
                
                image = cv2.resize(image, (640, 360))
                vid.write(image)
    except:
        pass
    cap.release()
    vid.release()
    save_data = np.array(save_data)
    np.save(video_path[:-4] + '_landmarks.npy', save_data)
    return save_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', help='path to the video file')

    args = parser.parse_args()
    mp_ground_truth(args.video)
