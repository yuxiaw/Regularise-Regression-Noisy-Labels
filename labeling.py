import os 
import random
import shutil
import logging
from scipy.stats import pearsonr, spearmanr
from util_io import save_txt, read_txt, get_sentences
from util_dataset import random_split_train_dev

logger = logging.getLogger(__name__)

def different_degree_corrupt(noise_ratios=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
    basedir="./data/N2C2/Noise/", num_group=11):
    for i in range(1, num_group):
        savedir = os.path.join(basedir, "noise_{}".format(i))
        pairs = read_txt(os.path.join(savedir, "train.txt"))
        N = len(pairs)
        for ratio in noise_ratios:
            name = "corrupt_"+str(ratio)[2:]+"/"
            corrupted_dir = os.path.join(savedir, name)
            if not os.path.exists(corrupted_dir):
                os.mkdir(corrupted_dir)
            
            num_noise = int(N*ratio)
            sampled_index = random.sample(range(N), num_noise)
            data, gold_labels, noisy_labels = [], [], []
            for j in range(N):
                if j in sampled_index:
                    line = pairs[j]
                    s1, s2, score = line.strip().split("\t")
                    noisy_label = round(random.uniform(0, 5), 2)
                    gold_labels.append(float(score))
                    noisy_labels.append(noisy_label)
                    noisy_line = s1 + "\t" + s2 + "\t" + str(noisy_label)
                    data.append(noisy_line)
                else:
                    data.append(pairs[j])
            pearson_corr = pearsonr(gold_labels, noisy_labels)[0]
            spearman_corr = spearmanr(gold_labels, noisy_labels)[0]
            print(corrupted_dir, pearson_corr, spearman_corr)
            logger.info("%s pearson_corr: %s, and spearman_corr: %s ", corrupted_dir, str(pearson_corr), str(spearman_corr))
            save_txt([str(k) for k in sampled_index], os.path.join(corrupted_dir, "train_index.txt"))
            save_txt(data, os.path.join(corrupted_dir, "train.txt"))
            shutil.copy(os.path.join(savedir, "dev.txt"), corrupted_dir)
            

def random_label_corrupt(filename="./data/N2C2/train.txt", savedir="./data/N2C2/Noise/", 
                        num_dev=400, num_noise=242, num_group=11):
    """Generate ten groups of train and dev saving savedir based on the data in filename
       Then corrupt num_noise labels in split train.txt"""
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    for i in range(1, num_group):
        random_split_train_dev(filename, os.path.join(savedir, "noise_{}".format(i)), num_dev)
        corrupted_dir = os.path.join(savedir, "noise_{}/corrupt/".format(i))
        if not os.path.exists(corrupted_dir):
            os.mkdir(corrupted_dir)
        pairs = read_txt(os.path.join(savedir, "noise_{}/train.txt".format(i)))
        sampled_index = random.sample(range(len(pairs)), num_noise)
        data, gold_labels, noisy_labels = [], [], []
        for j in range(len(pairs)):
            if j in sampled_index:
                line = pairs[j]
                s1, s2, score = line.strip().split("\t")
                noisy_label = round(random.uniform(0, 5), 2)
                gold_labels.append(float(score))
                noisy_labels.append(noisy_label)
                noisy_line = s1 + "\t" + s2 + "\t" + str(noisy_label)
                data.append(noisy_line)
            else:
                data.append(pairs[j])
        pearson_corr = pearsonr(gold_labels, noisy_labels)[0]
        spearman_corr = spearmanr(gold_labels, noisy_labels)[0]
        print(pearson_corr, spearman_corr)
        logger.info("pearson_corr: %s, and spearman_corr: %s ", str(pearson_corr), str(spearman_corr))
        save_txt([str(k) for k in sampled_index], os.path.join(corrupted_dir, "train_index.txt"))
        save_txt(data, os.path.join(corrupted_dir, "train.txt"))
        shutil.copy(os.path.join(savedir, "noise_{}/dev.txt".format(i)), corrupted_dir)

def resample_pairs(filename="./data/N2C2/train.txt", N=500, savedir="./data/N2C2/laymenlabel"):
    """leverage sentences of STS pairs, re-sample N pairs"""
    sentences, _ = get_sentences(filename)
    random.shuffle(sentences)
    data = []
    for i in range(N):
        s1 = random.choice(sentences)
        s2 = random.choice(sentences)
        data.append(s1 + "\t" + s2 + "\t")
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    save_txt(data, os.path.join(savedir, "train.txt"))

if __name__ == "__main__":
    # random_label_corrupt()
    different_degree_corrupt()

