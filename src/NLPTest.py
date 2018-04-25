import nltk

from nltk.metrics.scores import f_measure
from nltk.metrics.scores import precision
from nltk.metrics.scores import recall

import numpy as np
import matplotlib.pyplot as plt

import os
import collections


nltk.download('stopwords')

from sklearn.dummy import DummyClassifier

def split_and_add_label(X, y, criteriaIndex):
	documents = [] # An array of (words_in_document, classification)
	fileIndex = 0
	for filename in X:
		contents = str(filename.raw_text)
		print "fileIndex is: ", fileIndex
		split_contents = contents.split()
		documents.append((split_contents, y[fileIndex][criteriaIndex]))
		fileIndex += 1
	return documents

def bag_of_words(words):
	return dict([(word, True) for word in words])

# Plot the number of met/not met for each selection criterion
def visualize_scores(results, criterion, scores):
    """
    Inputs:
    results - list of lists of scores computed
    criteron - a list of criterion names
    scores - a list of names of scores to be computed

    Outputs:
    None (a bar chart)
    """

    N = 13
    ind = np.arange(N)  # the x locations for the groups
    width = 0.15       # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(ind, results[0], width, color='y')
    rects2 = ax.bar(ind + width, results[1], width, color='b')
    rects3 = ax.bar(ind + 2 * width, results[2], width, color='r')
    rects4 = ax.bar(ind + 3 * width, results[3], width, color='g')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Score')
    ax.set_title('Scores By Selection Criteria')
    ax.set_xticks(ind + width*3.0/2.0)
    ax.set_xticklabels(criterion)
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), scores, loc="best")

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                    '%.2f' % height,
                    ha='center', va='bottom', rotation=90)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    plt.xticks(rotation=90)
    plt.show()


def test(X, y):

	criterion = ["ABDOMINAL", "ADVANCED-CAD", "ALCOHOL-ABUSE", "ASP-FOR-MI", "CREATININE", "DIETSUPP-2MOS", "DRUG-ABUSE", "ENGLISH", "HBA1C", "KETO-1YR", "MAJOR-DIABETES", "MAKES DECISIONS", "MI-6MOS"]
	trainingTestSplitIndex = 161 	# 80% of 202 files, rounded down

	for criteriaIndex in range(3):
		documents = split_and_add_label(X, y, criteriaIndex)
		# Create and split data set
		print "creating datasets..."
		data_sets = [(bag_of_words(d), c) for (d, c) in documents]
		train_set, test_set = data_sets[:trainingTestSplitIndex], data_sets[trainingTestSplitIndex:]

		# Train on Naive Bayes Classifier. Print accuracy score and 
		# most informative features. 
		print "training..."
		classifier = nltk.NaiveBayesClassifier.train(train_set)
		print "for criteria ", criterion[criteriaIndex], " accuracy is: ", nltk.classify.accuracy(classifier, test_set)

		# bayes_classifier = nltk.NaiveBayesClassifier.train(train_set)
		# print nltk.classify.accuracy(bayes_classifier, test_set)
		# bayes_classifier.show_most_informative_features(10)

		# bayes_classifier = DummyClassifier.train(train_set)
		# print nltk.classify.accuracy(bayes_classifier, test_set)
		
		# bayes_classifier.show_most_informative_features(10)

		# dummyClf = DummyClassifier()
		# dummyClf.fit(train_set, test_set)

def test_v2(X, y):

	criterion = ["ABDOMINAL", "ADVANCED-CAD", "ALCOHOL-ABUSE", "ASP-FOR-MI", "CREATININE", "DIETSUPP-2MOS", "DRUG-ABUSE", "ENGLISH", "HBA1C", "KETO-1YR", "MAJOR-DIABETES", "MAKES DECISIONS", "MI-6MOS"]
	scores = ["ACCURACY", "F1", "PRECISION", "RECALL"]
	trainingTestSplitIndex = 161 	# 80% of 202 files, rounded down
	numFeatures = 3000 				# number of important words to keep

	results = [[] for s in scores]
	print "results: ", results
	
	#for criteriaIndex in [4, 5, 10]:
	for criteriaIndex in range(13):
		print "criterion: ", criterion[criteriaIndex]

		# taken from: https://pythonprogramming.net/naive-bayes-classifier-nltk-tutorial/
		all_words = []
		documents = []

		for i in range(len(X)):
			patient = X[i]
			completeFileText = []
			for entry in patient.entries:
				splitWords = entry.raw_text.split()
				completeFileText = completeFileText + splitWords
				all_words = all_words + entry.raw_text.split()
			patientTextAndLabel = (completeFileText, y[i][criteriaIndex])
			documents.append(patientTextAndLabel)

		all_words = nltk.FreqDist(all_words)
		word_features = list(all_words.keys())[:numFeatures]
		#word_features = list(all_words.keys())

		def find_features(document):
			words = set(document)
			features = {}
			for w in word_features:
				features[w] = (w in words)
			return features

		featuresets = [(find_features(rev), category) for (rev, category) in documents]
		training_set = featuresets[:trainingTestSplitIndex]
		testing_set = featuresets[trainingTestSplitIndex:]

		classifier = nltk.NaiveBayesClassifier.train(training_set)


		# Get scores: accuracy, f1, precision, recall
		refsets = collections.defaultdict(set)
		testsets = collections.defaultdict(set)

		# TODO: this is probably wrong, don't wanna split pos vs neg
		for i, (feats, label) in enumerate(training_set):
		    refsets[label].add(i)
		    observed = classifier.classify(feats)
		    testsets[observed].add(i)

		if (nltk.classify.accuracy(classifier, training_set) == None):
			results[0].append(0.0)
		else:
			results[0].append(nltk.classify.accuracy(classifier, testing_set))

		if (f_measure(refsets[1], testsets[1]) == None):
			results[1].append(0.0)
		else:
			results[1].append(f_measure(refsets[1], testsets[1]))

		if (precision(refsets[1], testsets[1]) == None):
			results[2].append(0.0)
		else:
			results[2].append(precision(refsets[1], testsets[1]))

		if (recall(refsets[1], testsets[1]) == None):
			results[3].append(0.0)
		else:
			results[3].append(recall(refsets[1], testsets[1]))

	print "results: ", results
	visualize_scores(results, criterion, scores)



		#classifier.show_most_informative_features(15)

		



	







