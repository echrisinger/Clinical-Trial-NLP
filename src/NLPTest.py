import nltk
import os
nltk.download('stopwords')

from sklearn.dummy import DummyClassifier

def poop(X, y, criteriaIndex):

	documents = [] # An array of (words_in_document, classification)

	# Loops through all the .xml files 
	# **** YOU WILL PROBABLY HAVE TO CHANGE THE FILE PATH BC/ IT'S HARDCODED ****
	fileIndex = 0
	for filename in X[:100]:
		contents = str(filename.raw_text)
		print "fileIndex is: ", fileIndex
		split_contents = contents.split()
		#print "split_contents", split_contents
		documents.append((split_contents, y[fileIndex][criteriaIndex]))
		fileIndex += 1
	return documents

def bag_of_words(words):
	return dict([(word, True) for word in words])

def test(X, y):

	criterion = ["ABDOMINAL", "ADVANCED-CAD", "ALCOHOL-ABUSE", "ASP-FOR-MI", "CREATININE", "DIETSUPP-2MOS", "DRUG-ABUSE", "ENGLISH", "HBA1C", "KETO-1YR", "MAJOR-DIABETES", "MAKES DECISIONS", "MI-6MOS"]

	for criteriaIndex in range(len(criterion)):
		documents = poop(X, y, criteriaIndex)
		# Create and split data set
		print "creating datasets..."
		data_sets = [(bag_of_words(d), c) for (d, c) in documents]
		train_set, test_set = data_sets[:len(data_sets)//2], data_sets[len(data_sets)//2:]

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












