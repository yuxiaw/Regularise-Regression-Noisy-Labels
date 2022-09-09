# coding=utf-8
# author: yuxia wang
# date: 28 Dec, 2020
# calculate average loss, r and p on train and validation set

import logging
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)

def mse_loss(preds, labels):
    assert(len(preds) == len(labels))
    N = len(preds)
    L = 0
    for i in range(N):
        L += pow((preds[i] - labels[i]), 2)
    average_loss = round(L/N, 4)
    return average_loss


def calculate_metrics(preds, labels, num_train=1242):
    assert(len(preds) == len(labels))
    assert(len(preds) > num_train)

    train_preds = preds[:num_train]
    train_labels = labels[:num_train]

    dev_preds = preds[num_train:]
    dev_labels = labels[num_train:]

    train_pearson_corr = round(pearsonr(train_preds, train_labels)[0], 4)
    train_spearman_corr = round(spearmanr(train_preds, train_labels)[0], 4)
    train_avg_loss = mse_loss(train_preds, train_labels)

    dev_pearson_corr = round(pearsonr(dev_preds, dev_labels)[0], 4)
    dev_spearman_corr = round(spearmanr(dev_preds, dev_labels)[0], 4)
    dev_avg_loss = mse_loss(dev_preds, dev_labels)

    return {
        "train_r": train_pearson_corr,
        "train_p": train_spearman_corr,
        "train_l": train_avg_loss,
        "dev_r": dev_pearson_corr,
        "dev_p": dev_spearman_corr,
        "dev_l": dev_avg_loss,
    }
