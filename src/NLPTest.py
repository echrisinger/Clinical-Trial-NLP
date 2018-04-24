import nltk
import os
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
	trainingTestSplitIndex = 161 	# 80% of 202 files, rounded down
	numFeatures = 1000 				# number of important words to keep
	
	for criteriaIndex in range(len(criterion)):
		print "criterion: ", criterion[criteriaIndex]

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

		# taken from: https://pythonprogramming.net/naive-bayes-classifier-nltk-tutorial/
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
		print "Naive Bayes Algorithm accuracy: ", nltk.classify.accuracy(classifier, testing_set)
		classifier.show_most_informative_features(15)



	







