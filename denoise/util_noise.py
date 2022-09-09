import os 
import random
import logging
import numpy as np
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)

def read_txt(filename):
    """Reads txt file into lines."""
    with open(filename) as file:
        lines = file.readlines()
        lines = [l.strip() for l in lines]
    return lines     

def save_txt(lines, savefile):
    """Save lines into txt file."""
    with open(savefile, "w") as writer:
        writer.write("\n".join(lines))

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

def get_scores(filename, score_pos = -1):
    """Reads similarity score from txt file."""
    lines = read_txt(filename)
    scores = []
    for line in lines:
        l = line.strip().split("\t")
        scores.append(round(float(l[score_pos]), 2))
    assert(len(scores) == len(lines))
    return scores

def get_pseudo_labels(pred_dir):
    files = os.listdir(pred_dir)
    preds = []
    for file in files:
        filename = os.path.join(pred_dir, file)
        # print(filename)
        lines = read_txt(filename)
        lines = [float(i) for i in lines]
        preds.append(lines)
    assert(len(preds) == len(files))  
    preds = np.array(preds).T
    print(preds.shape)
    N = preds.shape[0]

    pseudo_labels = []
    for i in range(N):
        avg = round(sum(preds[i])/len(preds[i]), 2)   
        pseudo_labels.append(avg)
    return pseudo_labels

def prediction_criterion(preds, golds, threshold = 0.75, noise_index = None):
    noise, clean = [], []
    for i in range(len(golds)): 
        if abs(preds[i]-golds[i]) > threshold:
            noise.append(i)
        else:
            clean.append(i)

    # print precision and recall after performing prediction criterion
    if noise_index is not None:
        overlap = [i for i in noise if i in noise_index]
        precision = round(len(overlap)/len(noise), 4)
        recall = round(len(overlap)/len(noise_index), 4)
        print(len(noise), len(overlap), precision, recall)
        logger.info("The number of detected noise: %s, overlap: %s, precision: %s, recall: %s", 
        str(len(noise)), str(len(overlap)), str(precision), str(recall))
    return noise, clean

def prediction_pearson_criterion(preds, golds, threshold = 0.75, 
    voters = 5, num_clean_subset = 8, threshold_r = 0.01, noise_index = None):
    """
    input: pseudo-labels based on predictions; annotated "gold" labels list; 
           index of noisy labels, to get the precision and recall (optional)
    return: detected noisy label index (and precision and recall)
    """
    # prediction criteria
    noise, clean = prediction_criterion(preds, golds, threshold, noise_index)
    # further apply pearson metric as criterion
    noise_r = []
    for j in noise:
        vote = []
        for k in range(voters):
            # randomly sample num_clean_subset exmaples from clean subsets
            R = random.sample(clean, num_clean_subset)
            P = [preds[i] for i in R]
            L = [golds[i] for i in R]
            r1 = pearsonr(P,L)[0]

            P += [preds[j]]
            L += [golds[j]]
            r2 = pearsonr(P,L)[0]
        
            # print(round(r1, 4), round(r2, 4))
            if abs(r1-r2) > threshold_r:
                vote.append(True)
            else:
                vote.append(False)
        if all(vote):
            noise_r.append(j)

    if noise_index is not None:
        overlap_r = [i for i in noise_r if i in noise_index]
        precision_r = round(len(overlap_r)/len(noise_r), 4)
        recall_r = round(len(overlap_r)/len(noise_index), 4)
        print(len(noise_r), len(overlap_r), precision_r, recall_r)
        logger.info("The number of detected noise: %s, overlap: %s, precision: %s, recall: %s", 
        str(len(noise_r)), str(len(overlap_r)), str(precision_r), str(recall_r))
    return noise_r

def identify_noisy_labels(pred_dir, train_data_dir, noise_detect_criterion = "prediction_pearson",
    threshold=0.75, voters = 5, num_clean_subset = 8, threshold_r = 0.01):
    """By judge the difference between mean of preds and gold labels
    if greater than threshold, 0.75 as default, will be considered as noisy labels
    and also consider the mean preds as pseudo-labels used in repairing"""
    pseudo_labels = get_pseudo_labels(pred_dir)
    gold = get_scores(os.path.join(train_data_dir, "train.txt"))
    assert(len(pseudo_labels) == len(gold))  

    # whether noise data index exist, if yes read into a list, otherwise set as None
    noise_index_dir = os.path.join(train_data_dir, "train_index.txt")
    if os.path.exists(noise_index_dir):
        noise_index = read_txt(noise_index_dir)
        noise_index = [int(i) for i in noise_index]
    else:
        print("noise label index does not exist.")
        noise_index = None
    
    if noise_detect_criterion == "prediction_pearson":
        noise = prediction_pearson_criterion(pseudo_labels, gold, threshold = threshold, 
        voters = voters, num_clean_subset = num_clean_subset, threshold_r = threshold_r, 
        noise_index = noise_index)
    else:
        noise, _ = prediction_criterion(pseudo_labels, gold, threshold = threshold,
        noise_index = noise_index)
    return noise, pseudo_labels

def update_train_data(pred_dir, train_data_dir, savetraindir, way_to_noise="discard", 
    noise_detect_criterion = "prediction_pearson", threshold = 0.75, 
    voters = 5, num_clean_subset = 8, threshold_r = 0.01):
    """Depending on the noise data index in train set
    keep all clean cases, and in three different way to noisy cases:
    discard: only save clean lines into updated train.txt
    resampling: to keep the same number of train cases, sample same number of clean cases to make up
    repair: save clean lines and noisy pairs with pseudo-labels
    # we assume that in train_data_dir, there are four files: 
    # train.txt: train set instances
    # train_index.txt: index of noisy labels in train.txt according to the index from 0-len(train)
    # dev_only.txt: the dev set instances
    # dev.txt: the merged version on train.txt and dev_only.txt"""
    noise, pseudo_labels = identify_noisy_labels(pred_dir, train_data_dir, 
    noise_detect_criterion = noise_detect_criterion, threshold = threshold, 
    voters = voters, num_clean_subset = num_clean_subset, threshold_r = threshold_r)
    print(noise)
    print("%d noisy labels in this epoch" % len(noise))
    logger.info("%d noisy labels in this epoch", len(noise))
    lines = read_txt(os.path.join(train_data_dir, "train.txt"))
    clean_lines = [lines[i] for i in range(len(lines)) if i not in noise]
    if way_to_noise == "discard":  
        lines_for_train = clean_lines
    elif way_to_noise == "pseudo":
        repair_lines = []
        for i, j in enumerate(noise):
            s1, s2, score = lines[j].strip().split("\t")
            pseudo = str(pseudo_labels[i])
            repair_lines.append(s1+"\t"+s2+"\t"+pseudo)
        assert(len(repair_lines) == len(noise))
        lines_for_train = clean_lines + repair_lines
    else:
        resample_lines = random.sample(clean_lines, len(noise))
        lines_for_train = clean_lines + resample_lines
    if not os.path.exists(savetraindir):
        os.mkdir(savetraindir)
    save_txt(lines_for_train, os.path.join(savetraindir, "train.txt"))
