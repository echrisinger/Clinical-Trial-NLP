import xml.etree.ElementTree as ET
import argparse
from os import walk
import numpy as np
import config

# Get the string paths of each file at the final directory of the path
def get_files(path='../res/train'):
    files = []
    for (dirpath, dirnames, filenames) in walk(path, topdown=False):
        print filenames
        # TODO: should have train_path at start
        files.extend([dirpath+'/'+name for name in filenames])
        break
    return files

def get_raw_words(files):
    X = []
    for i, f in enumerate(files):
        tree = ET.parse(f)
        root = tree.getroot()

        words = root.find('TEXT').text
        X.append(words)
    return X

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

    X = get_raw_words(files)
    y = get_labels(files)

if __name__ == "__main__":
    main()