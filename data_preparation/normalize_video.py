'''
Nomalize the landmarks based on an anchor frame.
Devansh Agarwal, da398@cornell.edu, 2/12/2023
'''

import cv2
import argparse
import numpy as np
import mediapipe as mp
from scipy.spatial.transform import Rotation as R

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_hands_detector = mp_hands.Hands(model_complexity=1,
            max_num_hands=1,min_detection_confidence=0.5, min_tracking_confidence=0.5)


def detect_hands_single_frame(image, hands_detector):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands_detector.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, # image to draw
            hand_landmarks, # model output
            mp_hands.HAND_CONNECTIONS, # hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    return image, results

def rotation_matrix_from_vectors_scipy(vec1List, vec2List):
    r = R.align_vectors(vec1List, vec2List)
    return r[0].as_matrix()

def normalize_body_landmarks(landmarks_path):
	landmarks = np.load(landmarks_path)
	res_landmarks = np.zeros(landmarks.shape)
	res_landmarks[:,0] = landmarks[:,0]
	for i in range(landmarks.shape[0]):
		orig_pose = landmarks[i][1:].reshape((-1,3))
		# find pelvis
		left_hip = orig_pose[5,:]
		right_hip = orig_pose[6,:]
		pelvis = (left_hip+right_hip)/2
		# center the pelvis as origin
		res_pose = orig_pose-pelvis
		# find the rotataion that rotates shoulders to pelvis to YZ plane
		left_shoulder = orig_pose[2,:]
		right_shoulder = orig_pose[1,:]
		p_ls = left_shoulder-pelvis
		p_rs = right_shoulder-pelvis
		normal = np.array([1,0,0])
		proj_p_ls = p_ls-normal*np.dot(p_ls, normal)
		proj_p_rs = p_rs-normal*np.dot(p_rs, normal)
		rot = R.align_vectors([proj_p_ls,proj_p_rs],[p_ls, p_rs])[0]
		# rotate the whole pose
		res_pose = rot.apply(res_pose)
		res_landmarks[i,1:]=res_pose.ravel()
	new_landmark_file_name = landmarks_path[:-4] + "_normalized.npy"
	np.save(new_landmark_file_name, landmarks)

def normalize_landmarks(landmarks_path, anchor_image_path, normalized_length):
	landmarks = np.load(landmarks_path)
	# print(landmarks[0])
	anchor_image = cv2.imread(anchor_image_path)

	annotated_image, mp_output = detect_hands_single_frame(anchor_image, mp_hands_detector)
	anchor_landmarks = mp_output.multi_hand_world_landmarks[0].landmark
	anchor_point0 = anchor_landmarks[0]
	anchor_point17 = anchor_landmarks[17]
	anchor_point5 = anchor_landmarks[5]

	anchor_vec_0_17 = np.array([anchor_point17.x - anchor_point0.x, anchor_point17.y - anchor_point0.y, anchor_point17.z - anchor_point0.z])
	anchor_vec_0_5 = np.array([anchor_point5.x - anchor_point0.x, anchor_point5.y - anchor_point0.y, anchor_point5.z - anchor_point0.z])
	anchor_vec_list= [anchor_vec_0_17, anchor_vec_0_5]
	for i in range(landmarks.shape[0]):
		coords = landmarks[i]
		coords_point0 = {"x":coords[1], "y": coords[2], "z": coords[3]}
		coords_point17 = {"x":coords[52], "y": coords[53], "z": coords[54]}
		coords_point5 = {"x":coords[16], "y": coords[17], "z": coords[18]}
		coords_vec_0_17 = np.array([coords_point17["x"] - coords_point0["x"], coords_point17["y"] - coords_point0["y"], coords_point17["z"] - coords_point0["z"]])
		coords_vec_0_5 = np.array([coords_point5["x"] - coords_point0["x"], coords_point5["y"] - coords_point0["y"], coords_point5["z"] - coords_point0["z"]])
		coords_vec_list= [coords_vec_0_17, coords_vec_0_5]
		rotation_matrix = rotation_matrix_from_vectors_scipy(anchor_vec_list, coords_vec_list)
		length_ratio = normalized_length/np.linalg.norm(coords_vec_0_17)
		wrist = np.array([coords_point0["x"], coords_point0["y"], coords_point0["z"]])
		# _frame_data = list()
		for j in range(1, 22):
			point = np.array([landmarks[i,j * 3 -2], landmarks[i,j * 3 -1], landmarks[i,j * 3]])
			new_point = rotation_matrix.dot(point - wrist) * length_ratio
			landmarks[i,j * 3 -2], landmarks[i,j * 3 -1], landmarks[i,j * 3] = new_point[0], new_point[1], new_point[2]
			# _frame_data.append(np.array([landmarks[i,j * 3 -2], landmarks[i,j * 3 -1], landmarks[i,j * 3]]))
		# data.append(_frame_data)
	new_landmark_file_name = landmarks_path[:-4] + "_normalized.npy"
	# print(landmarks, landmarks.shape)
	np.save(new_landmark_file_name, landmarks)
	# data = np.transpose(np.array(data))
	# anim = time_animate(data, mp_hands.HAND_CONNECTIONS)
	# writervideo = animation.FFMpegWriter(fps=30)
	# anim.save('./test_vis' +  new_landmark_file_name + '.mp4', writer=writervideo)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-lp', '--landmarks_path', help='path to the landmarks file')
    parser.add_argument('-aim', '--anchor_image', help='anchor frame for normalizing the landmarks')
    parser.add_argument('-l', '--length', type=float, help='length of the vector from point 0 to 17')


    args = parser.parse_args()
    normalize_landmarks(args.landmarks_path, args.anchor_image, args.length)
