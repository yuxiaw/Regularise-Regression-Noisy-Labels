# date 15 Jan 2021
# generate sentence pairs synthetically for non-medication topics

import os
import random
from util_io import read_txt, save_txt

def load_sentences():
    illnesslines = read_txt("./data/syn/sentence/illness.txt")
    instructionlines = read_txt("./data/syn/sentence/instruction.txt")
    diagnosislines = read_txt("./data/syn/sentence/diagnosis.txt")
    print(len(illnesslines), len(instructionlines), len(diagnosislines))

    topic_dict = dict()
    topic_dict["ill"] = illnesslines
    topic_dict["ins"] = instructionlines
    topic_dict["dia"] = diagnosislines
    return topic_dict

def synthetic_pairs(savedir="./data/syn/non_med_pairs.txt", 
                    num_zero = 1000, num_below = 1000, num_template = 3000):
    topic_dict = load_sentences()
    data = []
    # pair sents from different topic, score below 1.0
    for i in range(num_zero):
        t1, t2 = random.sample(["ins", "ill", "dia"], 2)
        s1 = random.choice(topic_dict[t1])
        s2 = random.choice(topic_dict[t2])
        tokens1 = s1.split()
        tokens2 = s2.split()
        overlap = [t for t in tokens1 if t in tokens2]
        r1 = round(len(overlap)/len(tokens1), 2)
        r2 = round(len(overlap)/len(tokens2), 2)
        r = round((r1+r2)/2, 1)
        data.append(s1 + "\t" + s2 + "\t" + str(r))

    # randomly pair from the same topic, more belew 2, in 1-2
    for i in range(num_below):
        t = random.choice(["ins", "ill", "dia"])
        s1 = random.choice(topic_dict[t])
        s2 = random.choice(topic_dict[t])
        tokens1 = s1.split()
        tokens2 = s2.split()
        overlap = [t for t in tokens1 if t in tokens2]
        r1 = round(len(overlap)/len(tokens1), 2)
        r2 = round(len(overlap)/len(tokens2), 2)
        r = round((r1+r2)/2, 1)
        if r == 1.0:
            data.append(s1 + "\t" + s2 + "\t" + "5.0")
        elif r >= 0.8:
            data.append(s1 + "\t" + s2 + "\t" + str(3.5+r))
        elif r >= 0.5:
            data.append(s1 + "\t" + s2 + "\t" + str(2.5+r))
        elif r >= 0.3:
            data.append(s1 + "\t" + s2 + "\t" + str(1.5+r))
        else:
            data.append(s1 + "\t" + s2 + "\t" + str(1.0+r))

    # template by replacing from another sentence of the same topic, 2-5
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    for i in range(num_template):
        t = random.choice(["ins", "ill", "dia"])
        s1 = random.choice(topic_dict[t])
        s = random.choice(topic_dict[t])
        tokens1 = s1.split()
        tokens = s.split()
        L = len(tokens1)
        m = min(L, len(tokens))
        s = random.choice(range(m))
        r = random.choice(ratios)
        gap = s + int(L*r)
        e = s + gap
        if e <= m:
            tokens2 = tokens1[:s] + tokens[s:e] + tokens1[e:]
            s2 = " ".join(tokens2)
            if r == 0.1:
                data.append(s1 + "\t" + s2 + "\t" + "4.5")
            elif r == 0.2:
                data.append(s1 + "\t" + s2 + "\t" + "4")
            elif r == 0.3:
                data.append(s1 + "\t" + s2 + "\t" + "3.5")
            elif r == 0.4:
                data.append(s1 + "\t" + s2 + "\t" + "3")
            else:
                data.append(s1 + "\t" + s2 + "\t" + "2")
    print(len(data)) 
    save_txt(data, savedir)  