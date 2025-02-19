'''
Generate a combined confusion matrix from several preds
7/5/2022, Ruidong Zhang, rz379@cornell.edu
'''

import argparse
import numpy as np

from libs.utils import generate_cm, load_gt, extract_labels

def combine_cm(pred_paths, output_path):
    combined_preds = []
    for p in pred_paths:
        combined_preds += load_gt(p)

    truths = [int(x[0]) for x in combined_preds]
    preds = [int(x[4]) for x in combined_preds]
    labels = extract_labels(combined_preds)

    generate_cm(np.array(truths), np.array(preds), labels, '%s%s' % ((output_path[:-4] if output_path[-4:].lower() == '.png' else output_path), '.png'))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Echo profile calculation')
    parser.add_argument('-p', '--pred', help='path to the prediction .txt file', action='append')
    parser.add_argument('-o', '--output', help='path to the output confusion matrix file')
    args = parser.parse_args()

    combine_cm(args.pred, args.output)