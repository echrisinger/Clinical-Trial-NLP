import argparse
import config
import numpy as np
import os
import re
import string 
import xml.etree.ElementTree as ET


from utils import *
from patient import *

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
        p_file = PatientFile(raw_words, entries=[Entry(text) for text in entries[:-1]])
        patient_files.append(p_file)
    return patient_files

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

def main():
    parser = argparse.ArgumentParser(description='Parse dataset params')
    parser.add_argument('train_path', type=str, help='path to the training data directory locally')
    args = parser.parse_args()

    train_path = args.train_path
    files = get_files(train_path)

    raw_X = get_raw_words(files)

    # break down the strings into individual files for a specific patient
    patient_files = group_entries(raw_X)
    pfiles_meta = get_metadata(patient_files)
    
    # scrub the punctuation for each individual file
    for p_i, p_file in enumerate(pfiles_meta):
        for i, entry in enumerate(p_file.entries):
            pfiles_meta[p_i].entries[i].text = scrub_punctuation(entry.raw_text)
    
    print pfiles_meta[0].entries[0].text
    y = get_labels(files)

if __name__ == "__main__":
    main()