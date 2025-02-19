'''
Load .raw or .wav audio file, and convert it to .wav if needed
'''
import wave
import argparse
import numpy as np

def load_audio(audio_file, original_bitwidth=16, target_bitwidth=16):
    assert(original_bitwidth == 16 or original_bitwidth <= 8)   # no point in using other bitwidths, I skipped them
    assert(2 <= target_bitwidth <= 16)

    if audio_file[-4:].lower() == '.raw':
        with open(audio_file, 'rb') as f:
            raw_file = f.read()
    else:
        wav = wave.open(audio_file, 'rb')
        num_frame = wav.getnframes()
        raw_file = wav.readframes(num_frame)

    if original_bitwidth == 16:
        decoded = np.frombuffer(raw_file, dtype=np.int16)
        converted = decoded
    elif original_bitwidth == 8:
        decoded = np.frombuffer(raw_file, dtype=np.int8).astype(np.int16) * 256
        converted = decoded
    else:
        raw = np.frombuffer(raw_file, dtype=np.uint8)
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
    
    if target_bitwidth < 16:
        dvd_val = (1 << (16 - target_bitwidth))
        downsampled = np.round(decoded.astype(float) / dvd_val)
    else:
        downsampled = converted.astype(float)
    return decoded, downsampled


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--audio', help='path to the audio file, .wav or .raw')
    parser.add_argument('-ob', '--original-bitwidth', help='original bitwidth, 16 or <= 8', type=int, default=16)
    parser.add_argument('-tb', '--target-bitwidth', help='target bitwidth, 2 - 16', type=int, default=16)
    # parser.add_argument('--save-wav', help='whether to save .wav file', action='store_true')
    parser.add_argument('-c', '--n-channels', help='number of channels', type=int, default=2)
    parser.add_argument('--fs', help='sampling rate to use for wav file', type=float, default=44100)


    args = parser.parse_args()
    _, raw = load_audio(args.audio, args.original_bitwidth, args.target_bitwidth)
    save_filename = args.audio[:-4] + '_%dto%d.wav' % (args.original_bitwidth, args.target_bitwidth)

    wavfile = wave.open(save_filename, 'wb')
    wavfile.setframerate(args.fs)
    wavfile.setsampwidth(2)
    wavfile.setnchannels(args.n_channels)
    wavfile.writeframes(raw.astype(np.int16).tobytes())
    wavfile.close()