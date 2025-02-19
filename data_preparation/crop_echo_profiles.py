import cv2

from data_preparation import load_config

parent_path = "../pilot_study/test_old_6/"
output_path = "cropped/obj5.png"
audio_num = "29"
echo_profile_type = "_fmcw_16bit_profiles.png"  # "_fmcw_16bit_diff_profiles.png"
start_frame = 1120
end_frame = 1210

echo_profile_img = cv2.imread(parent_path + "audio" + audio_num + echo_profile_type)
config = load_config(parent_path, '')
audio_config = config['audio']['config']

channel_height = round(echo_profile_img.shape[0] / 16)
echo_profile_img_width = echo_profile_img.shape[1]

crop_img = echo_profile_img[channel_height * 2 : channel_height * 4, start_frame:end_frame]
# round((video_ts[n_frames] - gt_syncing_pos) * audio_config['sampling_rate'] / audio_config['frame_length'] + audio_syncing_frame)
cv2.imwrite(parent_path + output_path, crop_img)