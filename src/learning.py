import nltk
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from dataset import get_Xy
from nltk.stem import *
from nltk.classify import maxent

from utils import stem_tokens
from NLPTest import bag_of_words
import config

def tfidf_tokenizer(text):
    tokens = nltk.word_tokenize(text)
    filtered = [w for w in tokens if not w in nltk.corpus.stopwords.words('english')]
    stems = stem_tokens(filtered, PorterStemmer())
    return stems

def partition_files():
    pfiles, y = get_Xy()
    X_train, X_test, y_train, y_test = train_test_split(pfiles, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def apply_tfidf(pfiles, y):
    tfidf_pos_1 = TfidfVectorizer(tokenizer=tfidf_tokenizer, stop_words='english')
    tfidf_neg_1 = TfidfVectorizer(tokenizer=tfidf_tokenizer, stop_words='english')
    arr_of_postfs = []
    arr_of_negtfs = []
    #config.NUM_LABELS
    for label in range(1):
        pos_pfiles, pos_y = zip(*[(pfile, y[i]) for i, pfile in enumerate(pfiles) if y[i][label] == 1])
        neg_pfiles, neg_y = zip(*[(pfile, y[i]) for i, pfile in enumerate(pfiles) if y[i][label] == -1])
        
        pos_text = [entry.text for entry in pfile.entries for pfile in pos_pfiles]
        arr_of_postfs += [tfidf_pos_1.fit_transform(pos_text)]

        neg_text = [entry.text for entry in pfile.entries for pfile in neg_pfiles]
        arr_of_negtfs += [tfidf_neg_1.fit_transform(neg_text)]
    return tfidf_pos_1, tfidf_neg_1

def apply_tfidf_24(pfiles, y, label_num):
    tfidf_pos_1 = TfidfVectorizer(tokenizer=tfidf_tokenizer, stop_words='english')
    tfidf_neg_1 = TfidfVectorizer(tokenizer=tfidf_tokenizer, stop_words='english')
    arr_of_postfs = []
    arr_of_negtfs = []
    #config.NUM_LABELS
    pos_pfiles, pos_y = zip(*[(pfile, y[i]) for i, pfile in enumerate(pfiles) if y[i][label_num] == 1])
    neg_pfiles, neg_y = zip(*[(pfile, y[i]) for i, pfile in enumerate(pfiles) if y[i][label_num] == -1])
    
    pos_text = [entry.text for entry in pfile.entries for pfile in pos_pfiles]
    arr_of_postfs += [tfidf_pos_1.fit_transform(pos_text)]
    
    neg_text = [entry.text for entry in pfile.entries for pfile in neg_pfiles]
    arr_of_negtfs += [tfidf_neg_1.fit_transform(neg_text)]
    return tfidf_pos_1, tfidf_neg_1

def top_tfidf_feats(row, features, top_n):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    top_names = [x[0] for x in top_feats]
    top_fracs = [x[1] for x in top_feats]
    sum_fracs = np.sum(top_fracs)
    return sum_fracs

def abdominal_tester(string):
    
    count = string.count("resection")
    if count > 1:
        return 1
    else:
        return 0

def diabetes_tester(string):
    count = string.count("neuropathy")
    count1 = string.count("retinopathy")
    count2 = count + count1
    if count2 > 1:
        return 1
    else:
        return 0


def diet_sup_tester(string):
    arr_of_words = ["vitamin", "folate", "supplement"]
    count = 0
    for x in arr_of_words:
        count1 = string.count(x)
        count += count1
    if count > 1:
        return 1
    else:
        return 0

def train_entropy_classifiers(pfiles, labels):
    train_toks_1 = []
    
    for x, person in enumerate(pfiles):
        test_text = ""
        test_entries = person.entries
        for y in test_entries:
            test_text += y.text
        words = bag_of_words(test_text)
        train_toks_1 += [(words, labels[x])]
    entropy_classifier = maxent.MaxentClassifier.train(train_toks_1)
    return entropy_classifier




        
def main():
    #X_train, X_test, y_train, y_test = partition_files()
    pfiles, y = get_Xy()
    tfidf_pos, tfidf_neg = apply_tfidf_24(pfiles, y, 0)

    #intuition gaining
    all_classifs = []
    classifications = []
    
    #TFIDF Testing
    '''
    for i in range(13):
        tfidf_pos, tfidf_neg = apply_tfidf_24(pfiles, y, i)
        for x in pfiles:
        
            test_entries = x.entries
            test_text = ""
            for i in test_entries:
                test_text += i.text
            response_pos = tfidf_pos.transform([test_text])
            response_neg = tfidf_neg.transform([test_text])

            response_pos_1 = np.squeeze(response_pos.toarray())
            response_neg_1 = np.squeeze(response_neg.toarray())
            pos_feature_names = tfidf_pos.get_feature_names()
            neg_feature_names = tfidf_neg.get_feature_names()
            pos_twenty = top_tfidf_feats(response_pos_1, pos_feature_names, 20)
            neg_twenty = top_tfidf_feats(response_neg_1, neg_feature_names, 20)
            if pos_twenty > neg_twenty:
                classifications += [1]
            else:
                classifications += [0]

        all_classifs += [classifications]
        classifications = []
    print all_classifs
    correct = 0.0
    accuracies = []
    for result in range(13):
        for elem in range(201):
            if all_classifs[result][elem] == y[elem][result]:
                correct += 1.0
        accuracy = correct / 201.0
        accuracies += [accuracy]
        correct = 0
    print accuracies
    '''



    #Abdominal Testing
    abdominal_predictions = []
    cad_predictions = []
    diet_predictions = []
    diab_predictions = []
    for pfile in pfiles:
        test_entries = pfile.entries
        
        test_text = ""
        
        for i in test_entries:
            test_text += i.text
        
        result_abdom = abdominal_tester(test_text)
#result_cad = cad_tester(test_text)
        result_diet = diet_sup_tester(test_text)
        result_diab = diabetes_tester(test_text)

        abdominal_predictions += [result_abdom]
#cad_predictions += [result_cad]
        diet_predictions += [result_diet]
        diab_predictions += [result_diab]
    abdom_accuracy = 0.0
    cad_accuracy = 0.0
    diet_accuracy = 0.0
    diab_accuracy = 0.0
    for x in range(len(y)):
        if y[x][0] == abdominal_predictions[x]:
            abdom_accuracy += 1.0
        #if y[x][0] == cad_predictions[x]:
        #cad_accuracy += 1.0
        if y[x][5] == diet_predictions[x]:
            diet_accuracy += 1.0
        if y[x][9] == diab_predictions[x]:
            diab_accuracy += 1.0
    
    
    X_train, X_test, y_train, y_test = partition_files()
    labels_0_train = []
    labels_1_train = []
    labels_2_train = []
    labels_3_train = []
    labels_4_train = []
    labels_5_train = []
    labels_6_train = []
    labels_7_train = []
    labels_8_train = []
    labels_9_train = []
    labels_10_train = []
    labels_11_train = []
    labels_12_train = []
    labels_0_test = []
    labels_1_test = []
    labels_2_test = []
    labels_3_test = []
    labels_4_test = []
    labels_5_test = []
    labels_6_test = []
    labels_7_test = []
    labels_8_test = []
    labels_9_test = []
    labels_10_test = []
    labels_11_test = []
    labels_12_test = []
    for result in y_train:
        labels_0_train += [result[0]]
        labels_1_train += [result[1]]
        labels_2_train += [result[2]]
        labels_3_train += [result[3]]
        labels_4_train += [result[4]]
        labels_5_train += [result[5]]
        labels_6_train += [result[6]]
        labels_7_train += [result[7]]
        labels_8_train += [result[8]]
        labels_9_train += [result[9]]
        labels_10_train += [result[10]]
        labels_11_train += [result[11]]
        labels_12_train += [result[12]]
    for result in y_test:
        labels_0_test += [result[0]]
        labels_1_test += [result[1]]
        labels_2_test += [result[2]]
        labels_3_test += [result[3]]
        labels_4_test += [result[4]]
        labels_5_test += [result[5]]
        labels_6_test += [result[6]]
        labels_7_test += [result[7]]
        labels_8_test += [result[8]]
        labels_9_test += [result[9]]
        labels_10_test += [result[10]]
        labels_11_test += [result[11]]
        labels_12_test += [result[12]]
    
    
    ent_classif_0 = train_entropy_classifiers(X_train, labels_0_train)
    ent_classif_1 = train_entropy_classifiers(X_train, labels_1_train)
    ent_classif_2 = train_entropy_classifiers(X_train, labels_2_train)
    ent_classif_3 = train_entropy_classifiers(X_train, labels_3_train)
    ent_classif_4 = train_entropy_classifiers(X_train, labels_4_train)
    ent_classif_5 = train_entropy_classifiers(X_train, labels_5_train)
    ent_classif_6 = train_entropy_classifiers(X_train, labels_6_train)
    ent_classif_7 = train_entropy_classifiers(X_train, labels_7_train)
    ent_classif_8 = train_entropy_classifiers(X_train, labels_8_train)
    ent_classif_9 = train_entropy_classifiers(X_train, labels_9_train)
    ent_classif_10 = train_entropy_classifiers(X_train, labels_10_train)
    ent_classif_11 = train_entropy_classifiers(X_train, labels_11_train)
    ent_classif_12 = train_entropy_classifiers(X_train, labels_12_train)

    train_toks_0 = []
    train_toks_1 = []
    train_toks_2 = []
    train_toks_3 = []
    train_toks_4 = []
    train_toks_5 = []
    train_toks_6 = []
    train_toks_7 = []
    train_toks_8 = []
    train_toks_9 = []
    train_toks_10 = []
    train_toks_11 = []
    train_toks_12 = []
    
    for x, person in enumerate(X_test):
        test_text = ""
        test_entries = person.entries
        for y in test_entries:
            test_text += y.text
            words = bag_of_words(test_text)
            train_toks_0 += [(words, y_test[x][0])]
            train_toks_1 += [(words, y_test[x][1])]
            train_toks_2 += [(words, y_test[x][2])]
            train_toks_3 += [(words, y_test[x][3])]
            train_toks_4 += [(words, y_test[x][4])]
            train_toks_5 += [(words, y_test[x][5])]
            train_toks_6 += [(words, y_test[x][6])]
            train_toks_7 += [(words, y_test[x][7])]
            train_toks_8 += [(words, y_test[x][8])]
            train_toks_9 += [(words, y_test[x][9])]
            train_toks_10 += [(words, y_test[x][10])]
            train_toks_11 += [(words, y_test[x][11])]
            train_toks_12 += [(words, y_test[x][12])]


    acc1 = nltk.classify.accuracy(ent_classif_0, train_toks_0)
    acc2 = nltk.classify.accuracy(ent_classif_1, train_toks_1)
    acc3 = nltk.classify.accuracy(ent_classif_2, train_toks_2)
    acc4 = nltk.classify.accuracy(ent_classif_3, train_toks_3)
    acc5 = nltk.classify.accuracy(ent_classif_4, train_toks_4)
    acc6 = nltk.classify.accuracy(ent_classif_5, train_toks_5)
    acc7 = nltk.classify.accuracy(ent_classif_6, train_toks_6)
    acc8 = nltk.classify.accuracy(ent_classif_7, train_toks_7)
    acc9 = nltk.classify.accuracy(ent_classif_8, train_toks_8)
    acc10 = nltk.classify.accuracy(ent_classif_9, train_toks_9)
    acc11 = nltk.classify.accuracy(ent_classif_10, train_toks_10)
    acc12 = nltk.classify.accuracy(ent_classif_11, train_toks_11)
    acc13 = nltk.classify.accuracy(ent_classif_12, train_toks_12)

    print [acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11, acc12, acc13]








#top_twenty = top_tfidf_feats(response, feature_names, 20)
    #print top_twenty
    #for col in response.nonzero()[1]:
#print feature_names[col], " - ", response[0, col]




if __name__ == "__main__":
    main()
