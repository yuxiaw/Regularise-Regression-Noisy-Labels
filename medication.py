# update 15 Jan 2021
# generate unique sentences of medication

import os
import random
from collections import Counter
from preprocess import tokenize
from util_io import read_txt, save_txt, get_sentences

def med_abbreviation_dict():
    abbr_dict = dict()
    abbr_dict["PO"] = ["taken orally", "taken through the mouth", "taken by mouth"]
    abbr_dict["po"] = ["taken orally", "taken through the mouth", "taken by mouth"]
    abbr_dict['p.o.'] = ["taken orally", "taken through the mouth", "taken by mouth"]
    abbr_dict['BID'] = ["twice a day", "twice daily"]
    abbr_dict['bid'] = ["twice a day", "twice daily"]
    abbr_dict['b.i.d.'] = ["twice a day", "twice daily"]
    abbr_dict['QID'] = ["four times a day"]
    abbr_dict['q.i.d.'] = ["four times a day"]
    abbr_dict['TID'] = ["three times a day"]
    abbr_dict['t.i.d.'] = ["three times a day"]
    abbr_dict['QD'] = ["once a day"]
    abbr_dict['qd'] = ["once a day"]
    abbr_dict['q.d.'] = ["once a day"]
    abbr_dict['q.h.s.'] = ["every bedtime"]
    abbr_dict['prn'] = ["as needed"]
    abbr_dict['prn.'] = ["as needed"]
    abbr_dict['PRN'] = ["as needed"]
    abbr_dict["Q72H"] = ["every 72 hours"]
    abbr_dict["Q48H"] = ["every 48 hours"]
    abbr_dict["Q24H"] = ["every 24 hours"]
    abbr_dict["Q10H"] = ["every 10 hours"]
    abbr_dict["Q8H"] = ["every 8 hours"]
    abbr_dict["Q6H"] = ["every 6 hours"]
    abbr_dict["Q4-6H"] = ["every four to six hours", "every 4-6 hours"]
    abbr_dict["Q4H"] = ["every 4 hours"]
    abbr_dict["Q2H"] = ["every 2 hours"]
    return abbr_dict

def clean_medication_sentence(sent):
    start = "Disp:*"
    end = "Refills:*"
    bracket_begin = "[**"
    bracket_end = "**]"
    while (start in sent and end in sent) or (bracket_begin in sent and bracket_end in sent):
        if start in sent and end in sent:
            s = sent.index(start)
            e = sent.index(end)
            temp = sent[:s] + sent[e+11:]
            temp = temp.strip()
            sent = temp
        else:
            if bracket_begin in sent and bracket_end in sent:
                s = sent.index(bracket_begin)
                e = sent.index(bracket_end)
                temp = sent[:s] + sent[e+3:]
                temp = temp.strip()
                sent = temp
    return sent.strip()

def generate_unique_medication_sentences(savedir="./data/syn/med_sents.txt", 
    filename = "./data/syn/Medications.txt", num=500):
    
    # get_medication_lines that has multiple sents as candidates
    lines = read_txt(filename)
    kws = ["1", "2", "3", "4", "5"]  # select lines has mutiple sents split by 1., 2. ...
    candidates = []
    for i, line in enumerate(lines[:num]):
        tokens = tokenize(line, False)
        if len([True for w in kws if w in tokens]) == 5:
            candidates.append(i)
    print("Among %d lines, we use %d, and %d are selected to be used." 
          % (len(lines), num, len(candidates)))

    # split into sentences according to 1., 2. position
    sents = []
    for i in candidates:
        line = lines[i]
        slots = []
        for pos in ["{}.".format(i) for i in range(1, 30)]:
            try:
                slots.append(line.index(pos))
            except ValueError:
                continue
        slots.append(len(lines))
        s = slots[0]
        for e in slots[1:]:
            sent = line[s:e]
            try:
                p = sent.index(" ")
                sents.append(sent[p+1:])
            except ValueError:
                sents.append(sent)
            s = e
    
    # keep a certain length sentence
    data = []
    for sent in sents:
        tokens = sent.split()
        if len(tokens) > 4 and len(tokens) < 25 and len(sent) > 20:
            data.append(sent.strip())
    # save_txt(data, savedir)

    # replace some abbreviation with their full name
    abbr_dict = med_abbreviation_dict()
    sentences = []
    for sent in data:
        tokens = sent.split()
        temp = []
        for token in tokens:
            if token in abbr_dict:
                temp.append(random.choice(abbr_dict[token]))
            else:
                temp.append(token)
        sentences.append(clean_medication_sentence(" ".join(temp)))
    
    # after clean, to further select
    final_data = []
    for sent in sentences:
        tokens = sent.split()
        if len(tokens) > 9:
            final_data.append(sent.strip())
    print(len(final_data))
    save_txt(final_data, savedir)
    return final_data

def generate_medication_pairs(filename, pairsavedir="./data/syn/med_pairs.txt", 
    num=2000, num_one=1000):
    sentences = generate_unique_medication_sentences(filename=filename, num=num)
    medication = dict()
    for i, sent in enumerate(sentences):
        med = sent.split()[0]
        if medication.get(med) is None:
            medication[med] = [i]
        else:
            medication[med] += [i]
    print("There are %d different medications." % len(medication.keys()))

    pairs = []
    for med in list(medication.keys()):
        sents = medication[med]
        N = len(sents)
        if N > 1:
            for i in range(N):
                i1, i2 = random.sample(sents, 2)
                s1 = sentences[i1]
                s2 = sentences[i2]
                t1 = s1.split()
                t2 = s2.split()
                overlap = [t for t in t1 if t in t2]
                r1 = round(len(overlap)/len(t1), 2)
                r2 = round(len(overlap)/len(t2), 2)
            if r1 == 1.0 and r2 == 1.0:
                pairs.append(s1 + "\t" + s2 + "\t" + "5.0")
            elif r1 >= 0.8 or r2 >= 0.8:
                pairs.append(s1 + "\t" + s2 + "\t" + "4.5")
            elif r1 >= 0.5 or r2 >= 0.5:
                pairs.append(s1 + "\t" + s2 + "\t" + "4.0")
            elif r1 <= 0.2 and r2 <= 0.2:
                pairs.append(s1 + "\t" + s2 + "\t" + "3")
            else:
                pairs.append(s1 + "\t" + s2 + "\t" + "3.5")
    print(len(pairs))

    for i in range(num_one):
        meds = list(medication.keys())
        m1,m2 = random.sample(meds,2)
        s1 = sentences[random.choice(medication[m1])]
        s2 = sentences[random.choice(medication[m2])]
        pairs.append(s1 + "\t" + s2 + "\t" + "1.0")
    print(len(pairs))
    save_txt(pairs, pairsavedir)

# ----------------------------------------------------------------
# Identify medication cases and statistical information
# ----------------------------------------------------------------
def is_medications(sent):
    """Given a raw sentence, identify whether 
    it's medication-related depending on key words"""
    key_words = ["mg", "ml", "tablet", "capsule"]
    tokens = tokenize(sent)
    for kw in key_words:
        if kw in tokens or any([(kw in token) for token in tokens]):
            return True
    return False

def medication_index(tokenlists):
    """Given a list of tokens, each tokens is from a sentence, return the index of
       medication sentence pairs"""
    key_words = ["mg", "ml", "tablet", "capsule"]
    med_index = []
    N = int(len(tokenlists)/2)
    for i in range(N):
        s1 = tokenlists[i*2]
        s2 = tokenlists[i*2+1]
        for kw in key_words:
            if kw in s1 and kw in s2:
                med_index.append(i)
            elif any([(kw in token) for token in s1]) and any([(kw in token) for token in s1]):
                med_index.append(i)     
    med_index = list(set(med_index))
    print(len(med_index))
    return med_index

def token_length(sentences, remove_stopwd=True):
    """calculate the averaged length of sentences, w/wt remove stopwords"""
    N = len(sentences)
    tokenlists = tokenize(sentences, remove_stopwd)
    lengths = sum([len(tokens) for tokens in tokenlists])
    return round(lengths/N, 2)

def medication_stats(filename, medsavedir=None, savename=None, savemed=True):
    """Given clinical STS data, return index of all medication cases
       and print statistical information: all data avg length (token-level)
       and medication cases score distribution
       parameter savemed is a bool value, means save medication cases or other cases"""
    sentences, scores = get_sentences(filename)
    print(token_length(sentences, False), token_length(sentences, True))
    tokenlists = tokenize(sentences)
    med = medication_index(tokenlists)
    med_scores = [scores[i] for i in med]
    counter = Counter(med_scores)
    print(counter)
    
    if medsavedir is not None and savename is not None:
        if not os.path.exists(medsavedir):
            os.mkdir(medsavedir)
        lines = read_txt(filename)
        if savemed:
            save_txt([lines[i].strip() for i in med], os.path.join(medsavedir, savename))
        else:
            save_txt([lines[i].strip() for i in range(len(lines)) if i not in med], os.path.join(medsavedir, savename))
    return med, counter

# ----------------------------------------------------------------
# Synthetic medication cases generation
# ----------------------------------------------------------------
def collect_medication_sentences(medfile):
    """collect unique medication sentences from exsiting medication pairs"""
    med_unique_sents = medfile[:-3]+"_unique_sents.txt"
    if os.path.exists(med_unique_sents):
        return read_txt(med_unique_sents)
    else:
        med_sents, _ = get_sentences(medfile)
        unique_med_sents = list(set(med_sents)) # cause random order each run due to set()
        print("Saving %d unique medication sentencs among %d sentences" % (len(unique_med_sents), len(med_sents)))
        save_txt(unique_med_sents, med_unique_sents)
        return unique_med_sents 

def find_name_dose_boundary(sent):
    """find the position spliting medicine name and dose, if fail to match key words
       return None, otherwise return character position and key word"""
    seg_words = ['tablet', 'capsule', 'solution', 'cream', 
             'packet', 'suspension', 'ointment', 'patch', 'kit', 'Liquid']
    for kw in seg_words:
        try:
            start = sent.index(kw)
            return start, kw
        except ValueError:
            continue
        return None
        
def get_medicine_name_dose(medfile):
    """extract medicine name, dose and taking frequency (dosage) for each sentence
       Input: medication txt file
       Return: three dictionary, value is the extracted name,dose and dosage"""
    unique_med_sents = collect_medication_sentences(medfile)
    med_dict = {}
    dosage_dict = {}
    frequency_dict = {}
    
    for i, sent in enumerate(unique_med_sents):
        ret = find_name_dose_boundary(sent)
        if ret is None:
            med_dict[i] = None
            dosage_dict[i] = None
            frequency_dict[i] = None
            # print("%d fails to search medicine name and dose" % i)
        else:
            start = ret[0]
            kw = ret[-1]
            med_dict[i] = sent[:start].strip().replace('"', '') + ' ' + kw
            try:
                end = sent.index("by mouth")
                dosage_dict[i] = sent[start+len(kw):end].strip()
                frequency_dict[i] = sent[end+8:]
            except ValueError:
                # print("%d dosage and frequency" % i)
                dosage_dict[i] = None
                frequency_dict[i] = None      
    return med_dict, dosage_dict, frequency_dict    
 
def analyse_dosage(dosage_dict):
    kind = []
    for i, dosage in dosage_dict.items():
        if dosage is not None:
            try:
                temp = dosage.split()[-2]
                if temp == 'mg' or temp == 'sustained':
                    print(dosage)
                else:
                    kind.append(temp)
            except:
                continue
    print(len(kind))
    return Counter(kind)

def analyse_frequency(frequency_dict):
    kind = []
    for i, fre in frequency_dict.items():
        if fre is not None:
            kind.append(fre)
    print(len(kind))
    return Counter(kind)

# ----------------------------------------------------------------
# Generation -- Pairing
# ----------------------------------------------------------------
def change_dosage(dosage):
    seq = ['0.5-1', '1', '1.5', 'one-half', '1-2', '2', '1-3', '1-4']
    if dosage is not None and dosage != '':
        temp = dosage.split()
        if temp[-2] in seq:
            seq.pop(seq.index(temp[-2]))
            temp[-2] = random.choice(seq)
            return ' ' + ' '.join(temp) + ' ' 
        else:
            temp[-2] = random.choice(['2.5', '3', '4', '5', '6', '6.5', '7', '7.5', '8', '8.5', '9', '9.5', '10'])
            return ' ' + ' '.join(temp) + ' ' 
    else:
        return dosage

def change_frequency(frequency):
    ending = ['as directed by prescriber as needed.', 'as directed by prescriber.', 'as directed as needed.',
          'before meals', 'after meal', 'as needed', '.']
    ones = ['one time', 'one time daily', 'one time a day', 'every 24 hours', 'every 24 hrs', 
                            'once a day', 'once daily']
    twos = ['two times', 'two times daily', 'two times a day', 'every 12 hours', 'every 12 hrs', 
                            'twice a day', 'twice daily']
    multis = ['three times a day', 'four times a day', 'every 4 hours', 'every 6 hours', 'every 8 hours',
                       'every 10 hours', 'every 2 hours']
    longs = ['every morning', 'every evening', 'every bedtime', 'every 48 hours', '']
    
    if 'one' in frequency:
        return ' ' + random.choice(twos+multis+longs) + ' ' + random.choice(ending)
    elif 'two' in frequency or 'twice' in frequency:
        return ' ' + random.choice(ones+multis+longs) + ' ' + random.choice(ending)
    else:
        return ' ' + random.choice(ones+twos+multis+longs) + ' ' + random.choice(ending)
        
def change_dose(medicine):
    ten = ['0.1', '0.4', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5', '5.5', '6', '8.6', '10']
    hundred = ['20', '24', '25', '30', '35', '40', '50', '60', '70', '75', '80', '81', '85', 
               '86', '90', '95', '100']
    thousand = ['500', '200', '220', '250', '120', '125', '150', '300', '325', '350', '400', '550', '600',
            '750', '800', '850', '900', '1000', '1,000', '2,000']
    temp = medicine.split()
    try:
        if temp[-3] in ten:
            ten.pop(ten.index(temp[-3]))
            temp[-3] = random.choice(ten)
        elif temp[-3] in hundred:
            hundred.pop(hundred.index(temp[-3]))
            temp[-3] = random.choice(hundred)
        elif temp[-3] in thousand:
            thousand.pop(thousand.index(temp[-3]))
            temp[-3] = random.choice(thousand)
        else:
            return None
    except IndexError:
        return None
    return ' '.join(temp) + ' '
    
def synonym(medicine):
    return medicine

def related_medicine(medicine):
    return medicine

def choose_different_sent(sent_index, unique_med_sents, med_dict):
    """Return a sentence from unique_med_sents, that is 
       1. differs from input sent_index
       2. differs from the medication of sent_index"""
    N = len(unique_med_sents)
    index = random.choice(range(N))
    while sent_index == index or med_dict[index] == med_dict[sent_index]:
        index = random.choice(range(N))
    return unique_med_sents[index]

def med_pairing(medfile, num_score_one=5, syndir=None):
    """Generate medication pairs based on the unique sentences in medfile by
    first extract component name, dose, dosage and taking frequency,
    then pairing and assign similarity scores by rules
    finally return all synthetically gnerated sentence pairs
    num_score_one: the number of pairs assigning 1.0 given a sentence"""
    unique_med_sents = collect_medication_sentences(medfile)
    med_dict, dosage_dict, frequency_dict = get_medicine_name_dose(medfile)
    pairs = []
    for i, sent in enumerate(unique_med_sents):
        # generate pairs labelling as 1
        time = 0
        while time < num_score_one:
            s1 = choose_different_sent(i, unique_med_sents, med_dict)
            pairs.append(random.choice([sent + "\t" + s1 + "\t" + "1.0", s1 + "\t" + sent + "\t" + "1.0"]))
            time += 1

        # change dosage and / or frequency
        seg1 = med_dict[i]
        seg2 = dosage_dict[i]
        seg3 = frequency_dict[i]
        if seg1 is not None and seg2 is not None and seg3 is not None:
            s45 = random.choice([seg1 + seg2 + ' by mouth' + change_frequency(seg3),
                                 seg1 + change_dosage(seg2) + 'by mouth' + seg3])
            s4 = seg1 + change_dosage(seg2) + 'by mouth' + change_frequency(seg3)
            pairs.append(random.choice([sent + "\t" + s45 + "\t" + "4.5", s45 + "\t" + sent + "\t" + "4.5"]))
            pairs.append(random.choice([sent + "\t" + s4 + "\t" + "4.0", s4 + "\t" + sent + "\t" + "4.0"]))

            # change medicine dose internally 
            seg1_alter = change_dose(seg1)
            if seg1_alter is not None:
                s3 = seg1_alter + change_dosage(seg2) + 'by mouth' + change_frequency(seg3)
                pairs.append(random.choice([sent + "\t" + s3 + "\t" + "3.0", s3 + "\t" + sent + "\t" + "3.0"]))
            else:
                continue
    print(len(pairs))
    if syndir is not None:
        if not os.path.exists(syndir):
            os.mkdir(syndir)
        save_txt(pairs, os.path.join(syndir, "syn_med_{}.txt".format(len(pairs))))
    return pairs

def sample_synthetic_med(syndir, N, basedir, file_to_merge=None):
    pairs = read_txt(syndir)
    sampled_pairs = random.sample(pairs, N)
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    save_txt(sampled_pairs, os.path.join(basedir, "synmed{}.txt".format(N)))
    if file_to_merge is not None:
        lines_to_merge = read_txt(file_to_merge)
        lines = sampled_pairs + lines_to_merge
    random.shuffle(lines)
    print(len(lines))
    save_txt(lines, os.path.join(basedir, "train.txt"))

if __name__ == "__main__":
    # med_dict, dosage_dict, frequency_dict = get_medicine_name_dose(
    #     "./data/N2C2/med/diff_cases_med.txt")
    # count1 = analyse_dosage(dosage_dict)
    # count2 = analyse_frequency(frequency_dict)
    # count3 = analyse_frequency(med_dict) 
    pairs = med_pairing("./data/N2C2/med/diff_cases_med.txt")