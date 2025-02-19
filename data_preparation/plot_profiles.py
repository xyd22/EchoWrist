import cv2
import argparse
import numpy as np
# from adaptive_static import adaptive_static


def plot_profiles(profiles, max_val=None, min_val=None):
    max_h = 0       # red
    min_h = 120     # blue
    if not max_val:
        max_val = np.max(profiles)
    if not min_val:
        min_val = np.min(profiles)
    # print(max_val, min_val)
    heat_map_val = np.clip(profiles, min_val, max_val)
    heat_map = np.zeros(
        (heat_map_val.shape[0], heat_map_val.shape[1], 3), dtype=np.uint8)
    # print(heat_map_val.shape)
    heat_map[:, :, 0] = heat_map_val / \
        (max_val + 1e-6) * (max_h - min_h) + min_h
    heat_map[:, :, 1] = np.ones(heat_map_val.shape) * 255
    heat_map[:, :, 2] = np.ones(heat_map_val.shape) * 255
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_HSV2BGR)
    return heat_map


def plot_profiles_split_channels(profiles, n_channels, maxval=None, minval=None):
    channel_width = profiles.shape[0] // n_channels

    profiles_img = np.zeros(
        ((channel_width + 5) * n_channels, profiles.shape[1], 3))

    for n in range(n_channels):
        channel_profiles = profiles[n * channel_width: (n + 1) * channel_width]
        profiles_img[n * (channel_width + 5): (n + 1) * (channel_width + 5) - 5,
                     :, :] = plot_profiles(channel_profiles, maxval, minval)

    return profiles_img


def plot_profiles_file(file_path, draw_diff, n_channels, maxval, maxdiff, mindiff):
    profiles = np.load(file_path)
    profiles_img = plot_profiles_split_channels(profiles, n_channels, maxval)
    cv2.imwrite(file_path[:-4] + '.png', profiles_img)
    if draw_diff:
        diff_profiles = profiles[:, 1:] - profiles[:, :-1]
        diff_profiles_img = plot_profiles_split_channels(
            diff_profiles, n_channels, maxdiff, mindiff)
        cv2.imwrite(file_path[:-13] + '_diff_profiles.png', diff_profiles_img)


# note: file_path should be the path to the raw files, as diff profiles may be generated easily from raw
def plot_alternating_channels(raw_profiles, n_channels, maxval=None, minval=None):
    diff_profiles_no_pad = raw_profiles[:, 1:] - raw_profiles[:, :-1]
    # diff profiles are one shorter, so must be padded
    padding = np.zeros((raw_profiles.shape[0], 1))
    diff_profiles = np.hstack((padding, diff_profiles_no_pad))
    both_profiles = np.vstack((raw_profiles, diff_profiles))

    channel_width = raw_profiles.shape[0] // n_channels
    profiles_img = np.zeros(
        ((channel_width + 5) * n_channels * 2, both_profiles.shape[1], 3))
    for n in range(n_channels):

        raw_channel_profile = raw_profiles[n *
                                           channel_width: (n + 1) * channel_width]
        diff_channel_profile = diff_profiles[n *
                                             channel_width: (n + 1) * channel_width]

        # we plot both channels in one iteration, so multiply index by 2
        raw_n = 2*n
        diff_n = 2*n + 1
        # 5 is added to channel_width is to leave black separating line btwn channels
        profiles_img[raw_n * (channel_width + 5): (raw_n + 1) * (channel_width + 5) - 5,
                     :, :] = plot_profiles(raw_channel_profile, maxval, minval)
        profiles_img[diff_n * (channel_width + 5): (diff_n + 1) * (channel_width + 5) - 5,
                     :, :] = plot_profiles(diff_channel_profile, maxval, minval)
    return profiles_img


# def draw_profiles_static(profiles, large_window, avg_window, n_avg_windows, n_channels=2):

#     print('Calculating static profiles')
#     static_profiles = adaptive_static(profiles, large_window, avg_window, n_avg_windows)
#     nostatic_profiles = profiles - static_profiles

#     print('Ploting static and nostatic profiless')
#     static_profiles_img = plot_profiles_split_channels(static_profiles, n_channels)
#     nostatic_profiles_img = plot_profiles_split_channels(nostatic_profiles, n_channels)

#     return static_profiles, static_profiles_img, nostatic_profiles_img

# def draw_profiles_static_with_path(profiles_path, large_window, avg_window, n_avg_windows, n_channels=2):
#     profiles = np.load(profiles_path)

#     static_profiles, static_profiles_img, nostatic_profiles_img = draw_profiles_static(profiles, large_window, avg_window, n_avg_windows, n_channels)

#     print('Saving files')
#     np.save(profiles_path[:-7] + 'static_profiles.npy', static_profiles)
#     # np.save(profiles_path[:-7] + '_simple_nostatic_profiles.png', nostatic_profiles_img)

#     cv2.imwrite(profiles_path[:-7] + 'static_profiles.png', static_profiles_img)
#     cv2.imwrite(profiles_path[:-7] + 'simple_nostatic_profiles.png', nostatic_profiles_img)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--profiles-file',
                        help='path to the .npy profiles file')
    parser.add_argument(
        '--diff', help='whether to output differential profiless', action='store_true')
    parser.add_argument('-n', '--n-channels',
                        help='number of channels', type=int, default=2)
    parser.add_argument(
        '-m', '--maxval', help='maxval for original profiles figure rendering, 0 for adaptive', type=int, default=8)
    parser.add_argument('-md', '--maxdiffval',
                        help='maxval for differential profiles figure rendering, 0 for adaptive', type=int, default=0.3)
    parser.add_argument('-nd', '--mindiffval',
                        help='maxval for differential profiles figure rendering, 0 for adaptive', type=int, default=-0.3)

    args = parser.parse_args()
    plot_profiles_file(args.profiles_file, args.diff, args.n_channels,
                       args.maxval, args.maxdiffval, args.mindiffval)
