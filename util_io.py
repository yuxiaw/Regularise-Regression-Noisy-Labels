# commonly-used functions for STS data in format of txt and tsv, including:
# read, save, get component elements: sentences and score, 
# calculate score distribution
import os
import csv
# ----------------------------------------------------------------
# Read and Save 
# ----------------------------------------------------------------
def read_tsv(input_file, delimiter = "\t", quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

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

def merge_files(filelist, savedir, name="train.txt"):
    lines = []
    for file in filelist:
        lines += read_txt(file)
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    save_txt(lines, os.path.join(savedir, name))

def get_sentences(filename, savename = None):
    """Reads STS txt file into sentences. If separate save is needed, pass savename directory"""
    sentences = []
    scores = []
    lines = read_txt(filename)
    for line in lines:
        s1, s2, score = line.split("\t")
        sentences.append(s1)
        sentences.append(s2)
        scores.append(score)
    sentences = [i.strip().replace("\n", " ").replace("\t", " ") for i in sentences]
    scores = [float(s.strip()) for s in scores]
    if savename is not None and (not os.path.exists(savename)):
        print("Saving sentences into a txt file...")
        with open(savename, "w") as writer:
            writer.write("\n".join(sentences))
    assert(len(sentences) == len(lines) * 2)
    assert(len(scores) == len(lines))
    return sentences, scores

def get_scores(filename, score_pos = -1):
    """Reads similarity score from txt file."""
    lines = read_txt(filename)
    scores = []
    for line in lines:
        l = line.strip().split("\t")
        scores.append(round(float(l[score_pos]), 2))
    assert(len(scores) == len(lines))
    return scores

# ----------------------------------------------------------------
# Score Distribution
# ----------------------------------------------------------------
def read_score_from_lines(lines, score_pos = -1):
    """Reads similarity score from lines."""
    scores = []
    for line in lines:
        l = line.strip().split("\t")
        scores.append(round(float(l[score_pos]), 2))
    assert(len(scores) == len(lines))
    return scores

def score_distribution(input_file):
    """Input labelled sts data, return the score distribution in a dict: (score interval: count)"""
    lines = read_txt(input_file)
    scores = read_score_from_lines(lines)

    intervals = {}
    keys = ['0-1.0', '1.1-2.0', '2.1-3.0', '3.1-4.0', '4.1-5.0']
    for k in keys:
        intervals[k] = 0

    for score in scores:
        if score <= 1.0:
            intervals['0-1.0'] = intervals['0-1.0'] + 1
        elif score <= 2.0:
            intervals['1.1-2.0'] = intervals['1.1-2.0'] + 1
        elif score <= 3.0:
            intervals['2.1-3.0'] = intervals['2.1-3.0'] + 1
        elif score <= 4.0:
            intervals['3.1-4.0'] = intervals['3.1-4.0'] + 1
        else:
            intervals['4.1-5.0'] = intervals['4.1-5.0'] + 1
    
    return intervals


def group_by_score(input_file):
    """Input labelled sts data, 
       return the score distribution in a list of list [[],[],...[]]
       each element list is the index of the instance """
    lines = read_txt(input_file)
    scores = read_score_from_lines(lines)

    data_distribution = [[],[],[],[],[]]
    for i, score in enumerate(scores):
        if score <= 1.0:
            data_distribution[0] += [i]
        elif score <= 2.0:
            data_distribution[1] += [i]
        elif score <= 3.0:
            data_distribution[2] += [i]
        elif score <= 4.0:
            data_distribution[3] += [i]
        else:
            data_distribution[4] += [i]
    
    return data_distribution, lines