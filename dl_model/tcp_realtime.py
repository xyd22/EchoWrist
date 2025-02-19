'''
Real-time system on the GPU machine's end for data reception and prediction
2/8/2022, Ruidong Zhang, rz379@cornell.edu
'''

import os
import cv2
import time
import wave
import socket
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from multiprocessing import Process, Queue
from matplotlib.animation import FuncAnimation

import torch
import torch.nn as nn
from libs.core import load_config
from libs.models import EncoderDecoder as ModelBuilder
from libs.utils import (AverageMeter, save_checkpoint, IntervalSampler,
                        create_optim, create_scheduler, print_and_log)

def tcp_listen(listen_port, pkt_length, q):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', listen_port))
    print('Socket bind complete')
    sock.listen(2)
    print('Socket now listening')
    (conn, (ip, port)) = sock.accept()
    print ('Got connection from ', (ip,port))
    buffer = b''
    while True:
        buffer += conn.recv(pkt_length)
        while len(buffer) > pkt_length:
            q.put(buffer[:pkt_length])
            buffer = buffer[pkt_length:]

def plot_CIR(cir, max_val=None, min_val=None):
    max_h = 0       # red
    min_h = 120     # blue
    if not max_val:
        max_val = np.max(cir)
    if not min_val:
        min_val = np.min(cir)
    # print(max_val, min_val)
    heat_map_val = np.clip(cir, min_val, max_val)
    heat_map = np.zeros((heat_map_val.shape[0], heat_map_val.shape[1], 3), dtype=np.uint8)
    # print(heat_map_val.shape)
    heat_map[:, :, 0] = heat_map_val / (max_val + 1e-6) * (max_h - min_h) + min_h
    heat_map[:, :, 1] = np.ones(heat_map_val.shape) * 255
    heat_map[:, :, 2] = np.ones(heat_map_val.shape) * 255
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_HSV2BGR)
    return heat_map

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_hm(seq, seq_len, maxval, minval):
    n_channels = seq.shape[1]
    # reshaped = np.reshape(seq[:, 0], (-1, 600))
    # print(seq.shape)
    reshaped = np.reshape(seq, (-1, seq_len, n_channels)).swapaxes(0, 2)
    reshaped = np.concatenate(reshaped, axis=0)
    # print(reshaped.shape)
    hm = plot_CIR(reshaped, maxval, minval)
    return hm

def update_plot(frame, im, q, seq_len, maxval, minval):
    last_frame_seq = None
    while not q.empty():
        last_frame_seq = q.get()    # T x n_channels
    
    # put the last frame back in case there is no frames at next update 
    q.put(last_frame_seq)
    hm = get_hm(last_frame_seq, seq_len, maxval, minval)
    im.set_data(hm)
    # p = plt.imshow(hm)
    # cv2.imwrite('test.png', hm)
    return im,

def continuous_plot(q, seq_len, maxval, minval):
    fig = plt.figure('CSI')
    ax = plt.subplot(1, 1, 1)
    # wait until images come in
    # fig, [ax1, ax2] = plt.subplots(1, 2)
    while q.empty():
        pass
    first_seq = q.get()
    q.put(first_seq)
    print(first_seq.shape)
    hm = get_hm(first_seq, seq_len, maxval, minval)
    p = plt.imshow(hm)
    ani = FuncAnimation(fig, update_plot, fargs=(p, q, seq_len, maxval, minval), interval=100, blit=True)
    plt.show()
    while True:
        pass

def cv2_plot(q, seq_len, maxval, minval):
    while q.empty():
        pass
    last_frame_seq = None
    while True:
        while not q.empty():
            last_frame_seq = q.get()    # T x n_channels
        # q.put(last_frame_seq)
        if last_frame_seq is not None:
            hm = get_hm(last_frame_seq, seq_len, maxval, minval)
            # print(hm.shape)
            cv2.imshow('CSI', hm)
            cv2.waitKey(30)

def load_tx():
    wav = wave.open('fmcw600_50k.wav', 'rb') # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes() # 获取帧数
    print(num_frame)
    n_channels = wav.getnchannels() # 获取声道数
    # print(n_channels)
    framerate = wav.getframerate() # 获取帧速率
    num_sample_width = wav.getsampwidth() # 获取实例的比特宽度，即每一帧的字节数
    str_data = wav.readframes(num_frame) # 读取全部的帧
    wav.close()
    wave_data = np.frombuffer(str_data, dtype = np.int16)
    # wave_data = np.reshape(wave_data, (-1, n_channels))
    # print(wave_data.shape, wave_data)
    return wave_data.astype(float)

def parse_pkt(raw_content, original_bitwidth):
    if original_bitwidth == 16:
        decoded = np.frombuffer(raw_content, dtype=np.int16)
    elif original_bitwidth == 8:
        decoded = np.frombuffer(raw_content, dtype=np.int8).astype(np.int16) * 256
    else:
        raw = np.frombuffer(raw_content, dtype=np.uint8)
        raw = raw[:raw.shape[0] - raw.shape[0] % original_bitwidth]
        raw = np.reshape(raw, (-1, original_bitwidth))
        converted = np.zeros((raw.shape[0], 8), dtype=np.int8)
        for p in range(8):
            start_bit_all = p * original_bitwidth
            end_bit_all = (p + 1) * original_bitwidth - 1
            start_bit_byte = start_bit_all // 8
            end_bit_byte = end_bit_all // 8
            combined_num = (raw.astype(np.uint16)[:, start_bit_byte] << 8) | raw[:, end_bit_byte]
            start_bit_in_combined_num = 8 - start_bit_all % 8 + 8
            end_bit_in_combined_num = start_bit_in_combined_num - original_bitwidth
            converted[:, p] = ((combined_num >> end_bit_in_combined_num) & ((1 << original_bitwidth) - 1)) << (8 - original_bitwidth)
        decoded = converted.ravel().astype(np.int16) * 256
    return decoded
    
def load_model(ckpt_path):
    config = load_config()  # load the configuration
    model = ModelBuilder(config['network'])  # load the designed model
    master_gpu = config['network']['devices'][0]
    model = model.cuda(master_gpu)  # load model from CPU to GPU
    # optimizer = create_optim(model, config['optimizer'])  # gradient descent
    model = nn.DataParallel(model, device_ids=config['network']['devices'])

    if os.path.isfile(ckpt_path):
        print_and_log('=> loading checkpoint {}'.format(ckpt_path))
        checkpoint = torch.load(ckpt_path,
                                map_location=lambda storage, loc: storage.cuda(master_gpu))
        # args.start_epoch = 0
        args.start_epoch = checkpoint['epoch']
        best_mae = checkpoint['best_mae']
        best_action_mae = checkpoint['best_action_mae']
        model.load_state_dict(checkpoint['state_dict'])
        # only load the optimizer if necessary
        print_and_log('=> loaded checkpoint {} (epoch {}, mae {:.3f}, action_mae {:.3f})'
                .format(ckpt_path, checkpoint['epoch'], best_mae, best_action_mae))
    else:
        print_and_log('=> no checkpoint found at {}'.format(ckpt_path))
        return

    return model

def pred(model, in_img):

    # expected shape: 1 x 87 x 200
    in_img = in_img[None, None, ...]
    in_img = in_img.astype(np.float32)
    input = torch.from_numpy(in_img)
    input = input.cuda()
    # print(input.shape)
    with torch.no_grad():
        output = model(input)
    return output.data.cpu().numpy()

def tcp_realtime(ckpt_path, listen_port, pkt_length, bitwidth, n_channels, plot_window, pred_window, pred_interval, stats_interval, pred_poi, fs=50000):

    tx_seq = load_tx()
    seq_len = tx_seq.shape[0]

    q_tcp = Queue()
    tcp_process = Process(target=tcp_listen, args=(listen_port, pkt_length, q_tcp))
    tcp_process.start()

    send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    send_sock.connect(('192.168.0.103', 6001))

    # q_plt = Queue()
    # plt_process = Process(target=cv2_plot, args=(q_plt, seq_len, None, None))
    # plt_process.start()

    all_rx_data = np.zeros((seq_len, 2), dtype=np.int16)
    bp_rx_data = np.zeros((seq_len, 2))
    corr = np.empty((0, n_channels))

    bp_start_idx = seq_len
    samples_plot_window = (round(plot_window * fs) // seq_len) * seq_len
    next_pred_time = pred_interval
    samples_pred_window = (round(pred_window * fs) // seq_len + 1) * seq_len

    t0 = 0
    next_display_time = stats_interval
    total_length = 0
    effective_length = 0
    lost_pkt = 0
    last_pkt_num = 0

    model = load_model(ckpt_path)
    model.eval()

    start_offset = -1

    try:
        while True:
            new_content = []
            while not q_tcp.empty():
                if t0 == 0:
                    t0 = time.time()
                new_data = q_tcp.get()
                total_length += len(new_data)
                effective_length += len(new_data) - 4
                pkt_idx = int.from_bytes(new_data[:4], 'big')

                # dealing with lost pkts
                if last_pkt_num == 0:
                    last_pkt_num = pkt_idx - 1
                lost_pkt += pkt_idx - last_pkt_num - 1
                if pkt_idx <= last_pkt_num:
                    print('Out of order: %d -> %d' % (last_pkt_num, pkt_idx))
                if pkt_idx - last_pkt_num - 1 > 0:
                    print('New pkt loss: %d' % (pkt_idx - last_pkt_num - 1))
                new_data_len = 0
                for _ in range(last_pkt_num + 1, pkt_idx):
                    new_content += [np.zeros((pkt_length - 4,), dtype=np.int16)]
                    # pkt_content = parse_pkt(np.zeros((pkt_length - 4,), dtype=np.uint8), bitwidth)
                    # pkt_content = np.reshape(pkt_content, (-1, n_channels))
                    # new_data_len += pkt_content.shape[0]
                    # all_rx_data = np.r_[all_rx_data, pkt_content]
                last_pkt_num = pkt_idx
                new_content += [parse_pkt(new_data[4:], bitwidth)]
                # parsing pkt
                # pkt_content = parse_pkt(new_data[4:], bitwidth)
                # pkt_content = np.reshape(pkt_content, (-1, n_channels))
                # new_data_len += pkt_content.shape[0]
                # all_rx_data = np.r_[all_rx_data, pkt_content]

            if len(new_content) == 0:
                continue
            new_content = np.concatenate(new_content)
            new_content = np.reshape(new_content, (-1, n_channels))
            all_rx_data = np.concatenate([all_rx_data, new_content], axis=0)
            this_bp_start_idx = (all_rx_data.shape[0] // seq_len - 1) * seq_len
            # not enough new data for updating
            if this_bp_start_idx <= 1 * seq_len or this_bp_start_idx == bp_start_idx:
                continue
            bp_piece = all_rx_data[bp_start_idx - seq_len: this_bp_start_idx]
            for c in range(n_channels):
                bp_piece[:, c] = butter_bandpass_filter(bp_piece[:, c], 15500, 20500, fs)
            bp_rx_data = np.r_[bp_rx_data, bp_piece[seq_len:]]
            bp_start_idx = this_bp_start_idx

            # bandpass and corr
            new_corr = []
            for c in range(n_channels):
                # channel_new_data = all_rx_data[-(new_data_len + seq_len - 1):, c].astype(float)
                channel_new_corr = np.correlate(bp_piece[1:, c], tx_seq)
                # channel_new_corr = channel_new_corr[seq_len - 1:]
                channel_new_corr = channel_new_corr[:bp_piece.shape[0] - seq_len]
                # print(channel_new_corr.shape, bp_piece.shape)
                new_corr += [channel_new_corr]
            new_corr = np.array(new_corr).T
            corr = np.r_[corr, new_corr]

            if start_offset < 0 and corr.shape[0] > 600 * 100:
                start_offset = np.argmax(np.abs(corr[:600 * 100, 0]))
                start_offset = (start_offset + 600 - 600 // 2) % 600
                print(start_offset)
                all_rx_data = all_rx_data[start_offset:, :]
                bp_rx_data = bp_rx_data[start_offset:, :]
                corr = corr[start_offset:, :]

            # plot
            # new_plot_windows = (corr.shape[0] - plot_start_idx) // samples_plot_window
            plot_start_idx = corr.shape[0] - samples_plot_window
            # print(corr.shape[0] - plot_start_idx, samples_plot_window, new_plot_windows)
            if plot_start_idx >= 0:
                # plot_start_idx += (new_plot_windows - 1) * samples_plot_window
                this_plot_window = corr[plot_start_idx: plot_start_idx + samples_plot_window]
                # put this data into a queue or send it out for visualization
                # q_plt.put(this_plot_window)

            # pred
            time_elapsed = time.time() - t0
            if time_elapsed >= next_pred_time:
                # print(corr.shape[0])
                pred_start_idx = corr.shape[0] - samples_pred_window - seq_len
                if pred_start_idx >= 0:
                    this_pred_window = corr[pred_start_idx: pred_start_idx + samples_pred_window + seq_len]
                    this_pred_window = np.reshape(this_pred_window, (-1, seq_len, n_channels)).swapaxes(0, 2)

                    diff_window = this_pred_window[:, :, 1:] - this_pred_window[:, :, :-1]
                    diff_window = diff_window[:, pred_poi[0]: pred_poi[1], :]
                    diff_window = np.concatenate(diff_window, axis=0).T

                    this_pred_window = this_pred_window[:, pred_poi[0]: pred_poi[1], :]
                    this_pred_window = np.concatenate(this_pred_window, axis=0).T
                    hm = plot_CIR(this_pred_window)
                    cv2.imwrite('test.png', hm)
                    preds = pred(model, diff_window)[0]
                    # print(preds)
                    send_str = ''
                    for bs in preds:
                        send_str += '%5.1f,' % (bs / 10)
                    send_str += '0'
                    send_sock.sendall(send_str.encode())
                next_pred_time += pred_interval
            # display_stats
            time_elapsed = time.time() - t0
            if t0 and time_elapsed >= next_display_time:
                raw_bps = total_length * 8 / time_elapsed
                effective_bps = effective_length * 8 / time_elapsed
                print('Time elapsed: %.1f, raw bps: %d, effective bps: %d, packet loss: %d, current pkt: %d' % (time_elapsed, round(raw_bps), round(effective_bps), lost_pkt, pkt_idx))
                next_display_time += stats_interval
    except Exception as e:
        print(e)
        tcp_process.terminate()
        tcp_process.join()
        # plt_process.terminate()
        # plt_process.join()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', help='checkpoint location')
    parser.add_argument('-p', '--port', help='listening port', type=int, default=9999)
    parser.add_argument('-l', '--pkt-length', help='packet length', type=int, default=244)
    parser.add_argument('-b', '--bitwidth', help='bitwidth', type=int, default=8)
    parser.add_argument('-n', '--n-channels', help='number of channels', type=int, default=2)
    parser.add_argument('-vw', '--visual-window', help='visualization window size in seconds', type=float, default=15)
    parser.add_argument('-pw', '--pred-window', help='prediction window size in seconds', type=float, default=1)
    parser.add_argument('-pi', '--pred-interval', help='prediction window size in seconds', type=float, default=0.033)
    parser.add_argument('-poi', '--pred-poi', help='prediction poi, e.g., 206,306', default='250,350')
    # parser.add_argument('-m', '--maxval', help='maxval for original CIR figure rendering, 0 for adaptive', type=int, default=8)
    # parser.add_argument('-md', '--maxdiffval', help='maxval for differential CIR figure rendering, 0 for adaptive', type=int, default=0.3)
    # parser.add_argument('-nd', '--mindiffval', help='maxval for differential CIR figure rendering, 0 for adaptive', type=int, default=-0.3)
    # args.add_argument('-o', '--output', default='temp', help='minimum area size')

    args = parser.parse_args()
    pred_poi = [int(x) for x in args.pred_poi.split(',')]
    tcp_realtime(args.ckpt, args.port, args.pkt_length, args.bitwidth, args.n_channels, args.visual_window, args.pred_window, args.pred_interval, 5, pred_poi)