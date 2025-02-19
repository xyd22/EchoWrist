'''
Truncate the 16-bit raw audio to 8 bit raw audio, so that SD-collected data can be used to train for BLE testing.
Ruidong Zhang, rz379@cornell.edu, 9/18/2023
'''

import argparse
import numpy as np

def truncate_8bit(source_audio, highest_bit, rounded=True):
    source_raw = np.frombuffer(open(source_audio, 'rb').read(), dtype=np.int16)
    target_raw = np.zeros(source_raw.shape, dtype=np.int8)

    target_raw = (((source_raw + ((1 << (highest_bit - 8 - 1)) if rounded else 0)) >> (highest_bit - 8)) & 0xff).astype(np.int8)

    # print(source_raw[:100])
    # print(target_raw[:100])

    with open(source_audio[:-4] + '_16to8_h%d.raw' % highest_bit, 'wb') as f:
        f.write(target_raw.tobytes())

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser('Truncate 16-bit raw audio to 8 bit')
    parser.add_argument('-a', '--audio', help='path to the prediction file')
    parser.add_argument('-hb', '--highest-bit', help='highest-bit', type=int, default=16)
    parser.add_argument('-r', '--rounded', help='rounded', action='store_true')

    args = parser.parse_args()

    truncate_8bit(args.audio, args.highest_bit, args.rounded)
