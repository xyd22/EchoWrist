'''
Data collection and labeling for acoustic silentspeech
2/18/2022, Ruidong Zhang, rz379@cornell.edu
'''

import os
import re
import cv2
import math
import time
import json
import random
import serial
import argparse
import numpy as np
from datetime import datetime
from playsound import playsound 
from gtts import gTTS

# TOCHANGE
from commands import load_cmds, generate_chi, generate_echowrist, generate_wristmotions, generate_chi_gesture, generate_flute


def load_config(config_path):
    if os.path.exists(config_path):
        config = json.load(open(config_path, 'rt'))
    else:
        config = json.load(open('demo_config.json', 'rt'))
        config['audio']['files'] = []
        config['audio']['syncing_poses'] = []
        config['ground_truth']['files'] = []
        config['ground_truth']['videos'] = []
        config['ground_truth']['syncing_poses'] = []
        config['sessions'] = []
    return config

def get_serial_port():
    all_dev = os.listdir('/dev')
    serial_ports = ['/dev/' + x for x in all_dev if x[:6] == 'cu.usb']
    if len(serial_ports) == 0:
        manual_serial_port = input('No serial port found, please specify serial port name: ')
        serial_ports += [manual_serial_port.rstrip('\n')]
    selected = 0
    if len(serial_ports) > 1:
        print('Multiple serial ports found, choose which one to use (0-%d)' % (len(serial_ports) - 1))
        for n, p in enumerate(serial_ports):
            print('%d: /dev/%s' % (n, p))
        selected = int(input())
    return serial_ports[selected]

def data_record(path_prefix, output_path, cmd_set, duration, folds, n_reps_per_fold, noserial, count_down, camera, audio, right_handed):
    if not os.path.exists(os.path.join(path_prefix, output_path)):
        print('Creating path', os.path.join(path_prefix, output_path))
        os.mkdir(os.path.join(path_prefix, output_path))

    config_path = os.path.join(path_prefix, output_path, 'config.json')
    config = load_config(config_path)

    cap = cv2.VideoCapture(camera)

    # right_camera = False
    # while not right_camera:
    #     cap = cv2.VideoCapture(1)
    #     if cap.get(cv2.CAP_PROP_FPS) < 10:
    #         right_camera = True
    #     else:
    #         cap.release()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame_size = (1280, 720)

    n_frames = 0
    uttered_words = 0
    valid_time_lapsed = 0
    # next_display = 5
    t0 = 0

    # cmds_1, imgs = load_cmds('music', folds, n_reps_per_fold)
    # cmds_t, imgs = load_cmds('int', folds, n_reps_per_fold)
    # cmds_2 = []
    # for cmd in cmds_t:
    #     cmds_2 += [(cmd[0] + 8, cmd[1], cmd[2])]
    # cmds = cmds_1 + cmds_2
    # cmds_isolated, imgs = load_cmds(cmd_set, folds, n_reps_per_fold)
    # cmds_connected, imgs = generate_connected_isolated_digits()
    # cmds = cmds_isolated# + [(-1, '', None, 5)] + cmds_connected
    # cmds = cmds_connected
    # cmds, imgs = generate_echowrist(folds, n_reps_per_fold)

    # TOCHANGE: change based on the gesture set
    cmds, imgs = generate_echowrist(folds, n_reps_per_fold)
    ts = []
    rcds = []

    save_pos = datetime.now().strftime('record_%Y%m%d_%H%M%S_%f')
    vid = cv2.VideoWriter(os.path.join(path_prefix, output_path, save_pos + '.mp4'), cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), 30, frame_size)

    audio_filename = None

    if not noserial:
        serial_port = get_serial_port()
        ser = serial.Serial(serial_port, 115200)
        print('Listening on', serial_port)
        ser.write(b's')
        print('Start signal sent')
        # ser.read(1)
        start_notice = ser.readline()
        if len(start_notice):
            filename_match = re.findall(r'(audio\d+\.raw)', start_notice.decode())
            if len(filename_match):
                audio_filename = filename_match[0]
                print('Audio filename: %s' % audio_filename)

    syncing_frame = 0
    syncing_ts = 0

    # Video command
    # num = 0
    # if duration == 1.5:
    #     num = 15
    # elif duration == 2.0:
    #     num = 20
    # elif duration == 2.5:
    #     num = 25
    # is_new_command = True
    # cap_command = cv2.VideoCapture("videos/Start.mov")

    cmd_idx = 0
    last_cmd_display_time = 0
    exit_flag = False
    syncing_received = False
    try:
    # if True:
        print('Session # %d' % (len(config['ground_truth']['files']) + 1))
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            if t0 == 0:
                t0 = time.time()
            n_frames += 1
            frame_ts = time.time()
            
            if not noserial and ser.inWaiting():
                in_bytes = ser.readline()
                # while in_bytes != b'000\n': pass
                if in_bytes == b'000\n':
                    syncing_received = True
                    syncing_frame = n_frames
                    syncing_ts = frame_ts
                    print('Received syncing signal, frame # %d, ts %.3f' % (syncing_frame, syncing_ts))
                    
            image = cv2.flip(image, 1)
            time_from_start = time.time() - t0
            info_text = 'Frame # %06d, ts: %.6f' % (n_frames, frame_ts)
            if time_from_start >= count_down and not exit_flag:
                info_text += ', %s' % cmds[cmd_idx][1]
            cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)
            vid.write(image)
            # image = cv2.resize(image, (5120, 2880))
            ts += [frame_ts]
            if not noserial and not syncing_received:
                print('Waiting for Teensy...')
                time.sleep(0.05)
                continue
            if time_from_start < count_down:
                cv2.putText(image, 'Please clap', (300, 380), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), thickness=3)
                cv2.putText(image, 'Session %d starting in %d s...' % (len(config['ground_truth']['files']) + 1, math.ceil(count_down - time_from_start)), (250, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), thickness=2)
            elif exit_flag and (time.time() -  session_end_time < count_down):
                cv2.putText(image, 'Please clap', (300, 380), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), thickness=3)
                cv2.putText(image, 'Session %d ending in %d s...' % (len(config['ground_truth']['files']) + 1, math.ceil(count_down - (time.time() - session_end_time))), (250, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), thickness=2)
                # Video command
                # cap_command = cv2.VideoCapture("videos/Start.mov")
            elif not exit_flag:
                if len(cmds[cmd_idx]) == 4:
                    duration = cmds[cmd_idx][3]

                image[600:, :] = 255
                # cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)
                if valid_time_lapsed == 0:
                    estimated_time_left = 0
                    wpm = 0
                    for i in range(cmd_idx, len(cmds)):
                        if len(cmds[cmd_idx]) == 4:
                            estimated_time_left += cmds[i][3]
                        else:
                            estimated_time_left += duration
                else:
                    estimated_time_left = (len(cmds) - cmd_idx) * valid_time_lapsed / cmd_idx
                    wpm = uttered_words / valid_time_lapsed * 60
                progress_text = 'Session progress: %.1f%%, current gpm: %.1f, estimated time left: %.1f s' % (cmd_idx / len(cmds) * 100, wpm, estimated_time_left)
                text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
                cv2.putText(image, progress_text, ((1280 - text_size) // 2, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)

                if time.time() - last_cmd_display_time > duration:
                    # display command
                    # print('Please touch area %s' % cmds[cmd_idx])
                    # cmd_idx += 1
                    if last_cmd_display_time > 0 and not bad_sample:
                        rcds += [(str(cmds[cmd_idx][0]), cmd_start_time, time.time(), cmds[cmd_idx][1])]
                        uttered_words += len(cmds[cmd_idx][1].split(' '))
                        valid_time_lapsed += rcds[-1][2] - rcds[-1][1]
                        cmd_idx += 1
                        # Audio command
                        if audio and cmd_set == 'objs' and cmd_idx < len(cmds):
                            language = 'en'
                            current = (cmds[cmd_idx])[1][-1]
                            save_text = "audios/" + current + ".wav"
                            if os.path.exists(save_text):
                                pass
                            else:
                                myobj = gTTS(text=current, lang=language, slow=False)
                                myobj.save(save_text)
                            playsound(save_text, block = False)
                        # Video command
                        # is_new_command = True
                    last_cmd_display_time = time.time()
                    cmd_start_time = time.time()
                    bad_sample = False
                elif last_cmd_display_time > 0:
                    cmd_progress = (time.time() - last_cmd_display_time) / duration
                    image[670:672, 0: round(cmd_progress * image.shape[1])] = 0
                    if cmds[cmd_idx][2] is not None:
                        inst_img = imgs[cmds[cmd_idx][2]]
                        if right_handed:
                            image[:inst_img.shape[0], -inst_img.shape[1]:, :] = inst_img[:, ::-1, :]
                        else:
                            image[:inst_img.shape[0], -inst_img.shape[1]:, :] = inst_img
                    # Video command
                    # if is_new_command:
                    #     is_new_command = False
                    #     cap_command = cv2.VideoCapture("videos/Gestures_" + str(num) + "/" + str(cmds[cmd_idx][1]) + ".mov")

                    if len(cmds[cmd_idx][1].split()) > 2:
                        text_1 = ' '.join(cmds[cmd_idx][1].split()[:3])
                        text_2 = ' '.join(cmds[cmd_idx][1].split()[3:])
                        text_size = cv2.getTextSize('%s' % text_1, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0][0]
                        cv2.putText(image, '%s' % text_1, ((1280 - text_size) // 2, 550), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), thickness=2)
                    else:
                        text_2 = cmds[cmd_idx][1]
                    text_size = cv2.getTextSize('%s' % text_2, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0][0]
                    cv2.putText(image, '%s' % text_2, ((1280 - text_size) // 2, 650), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), thickness=2)
                    
            else:
                break
            
            cv2.imshow('Data Collection', image)
            # cv2.setWindowProperty('Data Collection', cv2.WND_PROP_TOPMOST, 1)

            # Video command
            # success_command, image_command = cap_command.read()
            # if not success_command:
            #     image_command = np.zeros((337,600,3), dtype=np.uint8)
            #     print("Wrong Video")
            # flipped_image_command = cv2.flip(image_command, 1)
            # cv2.imshow('Command', flipped_image_command)
            

            pressed_key = cv2.waitKeyEx(1) & 0xFF
            # if pressed_key > 0:
            #     print(pressed_key)
            if not exit_flag:
                if cmd_idx == len(cmds):
                    session_end_time = time.time()
                    exit_flag = True
                if pressed_key == 27:
                    session_end_time = time.time()
                    exit_flag = True
                    # break
                if pressed_key == ord('x') or pressed_key == 2:
                    bad_sample = True
                    last_cmd_display_time = time.time() - duration
                    print('Bad:', frame_ts)
                if (pressed_key == ord(' ') or pressed_key == 3) and time.time() - last_cmd_display_time > 0.5:
                    last_cmd_display_time = min(last_cmd_display_time, time.time() - duration + 0.3)
    except:
        pass
    # time.sleep(duration)
    if not noserial:
        ser.write(b'e')
    cap.release()
    vid.release()
    # Video command
    # cap_command.release()
    with open(os.path.join(path_prefix, output_path, save_pos + '_records.txt'), 'wt') as f:
        for r in rcds:
            f.write('%s,%f,%f,%s\n' % r)
    
    # updating config file here
    if audio_filename:
        config['audio']['files'] += [audio_filename]
    config['ground_truth']['files'] += [os.path.basename(os.path.join(path_prefix, output_path, save_pos + '_records.txt'))]
    config['ground_truth']['videos'] += [os.path.basename(os.path.join(path_prefix, output_path, save_pos + '.mp4'))]
    config['sessions'] += [[]]

    # TOCHANGE: the number of the start of each fold
    fold_start_poses = [0, 10, 21, 41, 62, 70]
    fold_end_poses = [9, 19, 40, 60, 69, 77]
    # fold_start_poses = [0, 20, 41, 81, 122, 138]
    # fold_end_poses = [19, 39, 80, 120, 137, 153]

    # EchoWrist 2r, 2f
    # fold_start_poses = [0, 10, 21, 41, 62, 68]
    # fold_end_poses = [9, 19, 40, 60, 67, 73]

    # CHI 2r, 2f
    # fold_start_poses = [0, 6, 13, 19]
    # fold_end_poses = [5, 11, 18, 24]
    # fold_start_poses = [0, 30]
    # fold_end_poses = [29, 59]
    # fold_start_poses = [0, 14]
    # fold_end_poses = [13, 27]

    # General
    # gesture_num = len(cmds) / folds / n_reps_per_fold
    # fold_start_poses = []
    # fold_end_poses = []
    # for i in range(folds):
    #     fold_start_poses += [int(i * gesture_num * n_reps_per_fold)]
    #     fold_end_poses += [int((i + 1) * gesture_num * n_reps_per_fold)]
    print("num of cmds: ", len(cmds), "gesture num: ", gesture_num, "fold start poses: ", fold_start_poses)

    for fold_start_pos, fold_end_pos in zip(fold_start_poses, fold_end_poses):
        fold_end_pos = min(len(rcds) - 1, fold_end_pos)
        fold_start_ts = rcds[fold_start_pos][1]
        fold_duration = rcds[fold_end_pos][2] + 0.01 - fold_start_ts
        config['sessions'][-1] += [{
            'start': fold_start_ts,
            'duration': fold_duration
        }]
        if fold_end_pos >= len(rcds) - 1:
            break

    # for fold_start_pos in range(0, len(rcds), len(cmds) // folds):
    #     fold_end_pos = min(fold_start_pos + len(cmds) // folds, len(rcds)) - 1
    #     fold_start_ts = rcds[fold_start_pos][1]
    #     fold_duration = rcds[fold_end_pos][2] + 0.01 - fold_start_ts
    #     config['sessions'][-1] += [{
    #         'start': fold_start_ts,
    #         'duration': fold_duration
    #     }]
    json.dump(config, open(config_path, 'wt'), indent=4)
    with open(os.path.join(path_prefix, output_path, save_pos + '_frame_time.txt'), 'wt') as f:
        for t in ts:
            f.write('%f\n' % (t))
    # with open(os.path.join(path_prefix, output_path, 'CIR_syncing_frame.txt'), 'wt') as f:
    #     f.write('1,%d,%d\n' % (syncing_frame, int(syncing_ts)))
    if not noserial:
        print(ser.readline().decode())
        if ser.inWaiting():
            print(ser.readline().decode())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path-prefix', help='dataset parent folder', default='/Users/zrd/research_projects/echowrist/pilot_study')
    parser.add_argument('-o', '--output', help='output dir name')
    parser.add_argument('-c', '--commandsets', help='command set name, comma separated if multiple', default='15')
    parser.add_argument('-t', '--time', help='duration of each command/gesture', type=float, default=3)
    parser.add_argument('-f', '--folds', help='how many folds', type=int, default=1)
    parser.add_argument('-r', '--reps_per_fold', help='how many repetitiions per gesture for a fold', type=int, default=3)
    parser.add_argument('-cd', '--count_down', help='count down time (s) before start', type=int, default=3)
    parser.add_argument('--noserial', help='do not listen on serial', action='store_true')
    parser.add_argument('-cam', '--camera', help='the camera used for video capturing', type=int, default=0)
    parser.add_argument('--audio', help='toggle text to speech for commands', type = bool, default = False)
    parser.add_argument('--right', help='right handed', action='store_true')

    args = parser.parse_args()
    data_record(args.path_prefix, args.output, args.commandsets, args.time, args.folds, args.reps_per_fold, args.noserial, args.count_down, args.camera, args.audio, args.right)
