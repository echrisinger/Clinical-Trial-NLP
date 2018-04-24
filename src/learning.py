import nltk
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from dataset import get_Xy
from nltk.stem import *

from utils import stem_tokens
import config

def tfidf_tokenizer(text):
    tokens = nltk.word_tokenize(text)
    filtered = [w for w in tokens if not w in nltk.corpus.stopwords.words('english')]
    stems = stem_tokens(filtered, PorterStemmer())
    return stems

def partition_files():
    pfiles, y = get_Xy()
    X_train, X_test, y_train, y_test = train_test_split(pfiles, y, test_size=0.95)
    return X_train, X_test, y_train, y_test


def apply_tfidf(pfiles, y):
    tfidf = TfidfVectorizer(tokenizer=tfidf_tokenizer, stop_words='english')
    arr_of_postfs = []
    arr_of_negtfs = []
    #config.NUM_LABELS
    for label in range(2):
        pos_pfiles, pos_y = zip(*[(pfile, y[i]) for i, pfile in enumerate(pfiles) if y[i][label] == 1])
        neg_pfiles, neg_y = zip(*[(pfile, y[i]) for i, pfile in enumerate(pfiles) if y[i][label] == -1])
        
        pos_text = [entry.text for entry in pfile.entries for pfile in pos_pfiles]
        arr_of_postfs += [tfidf.fit_transform(pos_text)]

        neg_text = [entry.text for entry in pfile.entries for pfile in neg_pfiles]
        arr_of_negtfs += [tfidf.fit_transform(neg_text)]
    return tfidf

def top_tfidf_feats(row, features, top_n):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    top_names = [x[0] for x in top_feats]
    top_fracs = [x[1] for x in top_feats]
    print top_feats
    print top_names
    print top_fracs
        
def main():
    X_train, X_test, y_train, y_test = partition_files()
    tfidf = apply_tfidf(X_train, y_train)

    #intuition gaining
    test_one = X_train[0].raw_text
    response = tfidf.transform(test_one)
    print "original array", response
    response1 = response[0]
    print "shape is...", response.shape
    response_1 = np.squeeze(response1.toarray())
    feature_names = tfidf.get_feature_names()
    top_tfidf_feats(response_1, feature_names, 20)

#top_twenty = top_tfidf_feats(response, feature_names, 20)
    #print top_twenty
    #for col in response.nonzero()[1]:
#print feature_names[col], " - ", response[0, col]




if __name__ == "__main__":
    main()
