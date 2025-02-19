import os
import matplotlib
import numpy as np
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.font_manager import FontProperties

def Chinese_checker(s):
    for ch in s:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def generate_cm(truth, pred, labels, save_file):

    sns.set()
    is_Chinese = sum([Chinese_checker(l) for l in labels])
    if is_Chinese:
        matplotlib.rcParams['font.sans-serif'] = ['simhei']
        # sns.set_style({'font.sans-serif':['simhei']})
        # zh_font = FontProperties(fname=r'/usr/share/fonts/simhei.ttf')
        # sns.set(font=zh_font.get_family())
    # else:
    cm = confusion_matrix(truth, pred)
    # cm_1scale = deepcopy(cm)
    # np.savetxt(save_file + '_int.txt', cm)
    cm = (cm.T / np.sum(cm, 1)).T
    np.savetxt(save_file[:-4] + '_percentage.txt', cm)

    f, ax = plt.subplots(figsize=(cm.shape[0] // 2, cm.shape[0] // 2))
    acc = sum(pred == truth) / len(pred)
    idx_labels = [str(i) + ': ' + labels[i] for i in range(len(labels))]
    sns.heatmap(cm, cmap='Blues', annot=True, ax=ax, vmin=0, vmax=1, xticklabels=idx_labels, yticklabels=idx_labels, annot_kws={'size':6}, square=True)
    ax.set_title('Confusion matrix: acc = %.1f%%' % (100 * acc))
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Truth')
    f.savefig(save_file, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    truth = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 0, 0, 0, 0]
    pred = [1, 3, 3, 4, 1, 3, 3, 4, 1, 3, 3, 4, 1, 3, 3, 4, 0, 1, 1, 0]
    labels = ['00', '11', '22', '33', '44']
    generate_cm(truth, pred, labels, 'test_cm.png')