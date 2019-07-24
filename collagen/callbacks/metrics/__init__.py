import itertools
import re
from textwrap import wrap

import numpy as np
from matplotlib import figure
from sklearn.metrics import confusion_matrix

from ._meters import KappaMeter, SSBalancedAccuracyMeter, BalancedAccuracyMeter
from ._meters import Meter, RunningAverageMeter, AccuracyMeter, AccuracyThresholdMeter, SSAccuracyMeter, SSValidityMeter


def plot_confusion_matrix(correct_labels, predict_labels, labels, normalize=True):
    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    if normalize:
        cm = cm.astype('float') * 100.0 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = np.round(cm).astype('int')

    np.set_printoptions(precision=2)

    fig = figure.Figure(figsize=(5, 5), dpi=230, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=16)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=16, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=16)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=16, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=16,
                verticalalignment='center', color="black")
    fig.set_tight_layout(True)
    return fig