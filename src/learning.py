import nltk
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from dataset import get_Xy, parse_args
from nltk.stem import *
from sklearn.svm import SVC

from utils import stem_tokens, concat_entries
import config

def tfidf_tokenizer(text):
    tokens = nltk.word_tokenize(text)
    filtered = [w for w in tokens if not w in nltk.corpus.stopwords.words('english')]
    stems = stem_tokens(filtered, PorterStemmer())
    return stems

def apply_tfidf_label(pfiles, y, label=0):
    tfidf = TfidfVectorizer(tokenizer=tfidf_tokenizer, stop_words='english')
    
    # Get the labeled pfiles that are -1/1 for the label
    l_pfiles, y = zip(*[(pfile, y[i]) for i, pfile in enumerate(pfiles) if y[i][label] == 1 or y[i][label] == -1])
    # Get each patient's files into one big string, and make a list of them
    cat_lpfiles = concat_entries(l_pfiles)

    # Fit the tfidf
    tfs = tfidf.fit_transform(cat_lpfiles)
    return tfidf, tfs, y

def top_tfidf_feats(row, features, top_n):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    top_names = [x[0] for x in top_feats]
    top_fracs = [x[1] for x in top_feats]
    print top_feats
    print top_names
    print top_fracs
    
def main():
    args = parse_args()
    pfiles, y = get_Xy(args.train_path)
    X_train, X_test, y_train, y_test = train_test_split(pfiles, y, test_size=0.1)

    for label in range(args.n_labels):
        
        tfidf = TfidfVectorizer(tokenizer=tfidf_tokenizer, stop_words='english')
        
        # Get the labeled pfiles that are -1/1 for the label
        labeled_train_files, labeled_y = zip(*[(pfile, y_train[i][label]) for i, pfile in enumerate(X_train) if y_train[i][label] == 1 or y_train[i][label] == -1])
        labeled_y = list(labeled_y)
        for i in range(len(labeled_y)):
            if labeled_y[i] < 0:
                labeled_y[i] = 0

        # Get each patient's files into one big string, and make a list of them
        cat_train_files = concat_entries(labeled_train_files)
        cat_test_files  = concat_entries(X_test)

        cat_pfiles = concat_entries(pfiles)
        
        # Fit the tfidf
        tfidf.fit(cat_pfiles)
        tfs = tfidf.transform(cat_train_files).toarray()

        clf = SVC(C=1.0, kernel='rbf')
        clf.fit(tfs, labeled_y)
        
        test_labels = [y_test[i][label] for i in range(len(y_test))]
        test_tfs    = tfidf.transform(cat_test_files).toarray()

        for i in range(len(test_labels)):
            if test_labels[i] < 0:
                test_labels[i] = 0
                
        acc = clf.score(test_tfs, test_labels)
        print 'label {} accuracy is {}'.format(label, acc)


if __name__ == "__main__":
    main()
