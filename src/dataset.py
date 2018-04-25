import argparse
import config
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import string 
import xml.etree.ElementTree as ET

import NLPTest

from utils import *
from patient import *

column_titles = ["ABDOMINAL", "ADVANCED-CAD", "ALCOHOL-ABUSE", "ASP-FOR-MI", "CREATININE", "DIETSUPP-2MOS", "DRUG-ABUSE", "ENGLISH", "HBA1C", "KETO-1YR", "MAJOR-DIABETES", "MAKES-DECISIONS", "MI-6MOS"]

# Get the string paths of each file at the final directory of the path
def get_files(path='../res/train'):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path, topdown=False):
        # TODO: should have train_path at start
        files.extend([dirpath+'/'+name for name in filenames])
        break
    return files

# get the raw string associated with each individual
def get_raw_words(files):
    X = []
    for i, f in enumerate(files):
        tree = ET.parse(f)
        root = tree.getroot()

        words = root.find('TEXT').text
        X.append(words)
    return X

# group the raw string's contained entries into a string per entry
def group_entries(raw_words):
    patient_files = []
    for f_words in raw_words:
        entries = f_words.split(config.FILE_DELIMITER)
        p_file = PatientFile(f_words, entries=[Entry(text) for text in entries[:-1]])
        patient_files.append(p_file)
    return patient_files

def get_train_path():
    parser = argparse.ArgumentParser(description='Parse dataset params')
    parser.add_argument('train_path', type=str, help='path to the training data directory locally')
    args = parser.parse_args()
    return args.train_path

def get_Xy():
    train_path = get_train_path()
    files = get_files(train_path)

    raw_X = get_raw_words(files)

    # break down the strings into individual files for a specific patient
    patient_files = group_entries(raw_X)
    pfiles_meta = get_metadata(patient_files)
    
    # scrub the punctuation for each individual file
    for p_i, p_file in enumerate(pfiles_meta):
        for i, entry in enumerate(p_file.entries):
            pfiles_meta[p_i].entries[i].text = scrub_punctuation(entry.raw_text).lower()

    y = get_labels(files)
    return pfiles_meta, y

def vectorize_files(patient_files):
    # add all the words to a set and sort it
    words = set()
    [words.add(w) for w in re.split('s+', f) for f in p_files for p_files in patient_files]
    words = list(words)
    words.sort()

    for p_files in patient_files:
        for f in p_files:
            f_words = re.split('s+', f)
            f_words.sort()

# Get the labels of n files, returned in an n-length array.
# Uses globally defined constant for number of labels
def get_labels(files):
    y = np.ndarray((len(files), config.NUM_LABELS))
    for i, f in enumerate(files):
        tree = ET.parse(f)
        root = tree.getroot()

        labels = root.find('TAGS')
        y_f = np.zeros(len(labels))
        for j, label in enumerate(labels):
            if (label.attrib['met'] == 'not met'):
                y_f[j] = -1
            elif (label.attrib['met'] == 'met'):
                y_f[j] = 1

        y[i] = y_f
    return y


# Count up the number of met/not met for each selection criterion
def met_not_met_counts(y):
    """
    Inputs:
    y -- the labels, a (202, 13) array

    Outputs:
    label_sums -- (13, 2) array, each row is a selection criterion,
                    column 0 is "not met" counts, column 1 is "met" counts
    """
    label_sums = np.zeros((13, 2))
    for labels in y:
        for j in range(13):
            if labels[j] < 0:
                label_sums[j][0] = label_sums[j][0] + labels[j]
            if labels[j] > 0:
                label_sums[j][1] = label_sums[j][1] + labels[j]
    return label_sums


# Plot the number of met/not met for each selection criterion
def visualize_original_labels(label_sums):
    """
    Inputs:
    label_sums -- (13, 2) array, each row is a selection criterion,
                    column 0 is "not met" counts, column 1 is "met" counts
    Outputs:
    None (a bar chart)
    """
    met = []
    not_met = []
    for i in label_sums:
        met.append(i[1])
        not_met.append(i[0] * -1)

    N = 13
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, met, width, color='y')
    rects2 = ax.bar(ind + width, not_met, width, color='b')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Count')
    ax.set_title('Met/Not Met Counts By Selection Criteria')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(column_titles)
    ax.legend((rects1[0], rects2[0]), ('Met', 'Not Met'))

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    plt.xticks(rotation=90)
    plt.show()

def main():
    X, y = get_Xy()
    # print "X is: ", X[0].raw_text
    # print "y is: ", y

    NLPTest.test(X, y)

    # Bar chart of met/not met counts for each selection criterion
    label_sums = met_not_met_counts(y)
    visualize_original_labels(label_sums)


if __name__ == "__main__":
    main()
