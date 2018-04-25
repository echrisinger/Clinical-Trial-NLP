import nltk
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from dataset import get_Xy, parse_args, column_titles
from nltk.stem import *
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


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

def main():
    args = parse_args()
    pfiles, y = get_Xy(args.train_path)
    totals = {}
    totals['C'] = 1
    totals['accuracy'] = [0 for _ in range(13)]
    totals['f1']  = [0 for _ in range(13)]
    totals['auroc'] = [0 for _ in range(13)]
    totals['precision'] = [0 for _ in range(13)]
    totals['sensitivity'] = [0 for _ in range(13)]
    totals['specificity'] = [0 for _ in range(13)]

    f_svc = open('svc_results.txt', "a+")
    trials = 2
    for k in range(trials):
        X_train, X_test, y_train, y_test = train_test_split(pfiles, y, test_size=0.9)
        label_scores = [{} for _ in range(config.NUM_LABELS)]
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

            c_range = [1]
            c_scores = [0 for _ in c_range]
            
            f_svc.write('label #: {}'.format(column_titles[label]))
            clf = SVC(C=1, kernel='rbf')
            if not labeled_y.__contains__(0):
                continue
            clf.fit(tfs, labeled_y)
            
            test_labels = [y_test[k][label] for k in range(len(y_test))]
            test_tfs    = tfidf.transform(cat_test_files).toarray()

            for j in range(len(test_labels)):
                if test_labels[j] < 0:
                    test_labels[j] = 0
                    
            results = clf.predict(test_tfs)

            totals['C'] = 1
            totals['accuracy'][label] += clf.score(test_tfs, test_labels)
            totals['f1'][label]  += f1_score(test_labels, results)
            totals['auroc'][label] += roc_auc_score(test_labels, results, average="macro")
            totals['precision'][label] += precision_score(test_labels, results)
            totals['sensitivity'][label] += recall_score(test_labels, results)
            totals['specificity'][label] += (lambda tn, fp, fn, tp: tn / float(tn+fp))(*(confusion_matrix(test_labels, results).ravel()))

    print("C: 1\n")

    totals['accuracy']  = [x/trials for x in totals['accuracy']]
    totals['f1'] = [x/trials for x in totals['f1']]
    totals['auroc'] = [x/trials for x in totals['auroc']]
    totals['precision'] = [x/trials for x in totals['precision']]
    totals['sensitivity'] = [x/trials for x in totals['sensitivity']]
    totals['specificity']  = [x/trials for x in totals['specificity']]

    print("accuracy: {}\n".format(totals['accuracy']))
    print("f1: {}\n".format(totals['f1']))
    print("auroc: {}\n".format(totals['auroc']))
    print("precision: {}\n".format(totals['precision']))
    print("sensitivity: {}\n".format(totals['sensitivity']))
    print("specificity: {}\n\n".format(totals['specificity']))
        
    f_svc.write("C: {}\n".format(1))
    f_svc.write("accuracy: {}\n".format(totals['accuracy']))
    f_svc.write("f1: {}\n".format(totals['f1']))
    f_svc.write("auroc: {}\n".format(totals['auroc']))
    f_svc.write("precision: {}\n".format(totals['precision']))
    f_svc.write("sensitivity: {}\n".format(totals['sensitivity']))
    f_svc.write("specificity: {}\n\n".format(totals['specificity']))
        
    f_svc.close()

if __name__ == "__main__":
    main()
