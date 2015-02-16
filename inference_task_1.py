from sklearn import cross_validation
import cPickle as pickle
import sys
import numpy as np

import models
from data import *

def measure_error(train, test, classifier):

	classifier.train(train)
	predictions = classifier.predict(test)

	test_labels = [np.mean(filter(None,user.get_spendings())) for user in test]

	error =  sum([np.abs(x-y)**2 for x,y in zip(predictions,test_labels)])

	print len(test_labels)
	print error 
	return np.sqrt(error) / np.sqrt(len(test_labels)*9)

def first_inference(full_data, classifier, n_folds=10):
	# exclude users with no price-tag
	temp_data = []
	for user in full_data:
		if not filter(None,user.get_spendings()):
			continue
		else:
			temp_data.append(user)

	full_data = temp_data

	folds = cross_validation.KFold(len(full_data),n_folds=n_folds)

	rmse = 0

	for train, test in folds:
		trainset = [full_data[i] for i in train]
		testset = [full_data[i] for i in test]
		rmse += measure_error(trainset, testset, classifier)

	return (rmse / float(len(folds)))

def get_venue_labels(dataset, venue_types):
	""" If user has visited venue_type, label = 1, otherwise = 0 """

	venue_labels = []

	for user in dataset:
		user_visited = 0
		for checkin in user.foursquare:
			for venue_type in venue_types:
				if venue_type in checkin.lowest_type:
					user_visited = 1
					break
				else:
					pass
		if user_visited == 1:
			venue_labels.append(1)
		else:
			venue_labels.append(0)

	return venue_labels

def measure_venue_error(train, test, trainlabels, testlabels, classifier):
	""" Measure recall and precision, true negative rate, accuracy """
	classifier.train(train, trainlabels)
	predictions = classifier.predict(test)
	
	false_positives = 0
	false_negatives = 0
	true_positives = 0
	true_negatives = 0

	for i in range(len(predictions)):
		if predictions[i] == 1 and testlabels[i] == 0:
			false_positives += 1
		elif predictions[i] == 0 and testlabels[i] == 1:
			false_negatives += 1
		elif predictions[i] == 1 and testlabels[i] == 1:
			true_positives += 1
		elif predictions[i] == 0 and testlabels[i] == 0: #for sake of readability
			true_negatives += 1 
		else:
			pass

	error = false_negatives + false_positives

	try:
		recall = true_positives/float(true_positives + false_negatives)
	except:
		recall = np.nan
	try:
		precision = true_positives/float(true_positives+false_positives)
	except:
		precision = np.nan
	truenegrate = true_negatives/float(true_negatives+false_positives)
	accuracy = (true_positives + true_negatives) / float(true_positives+true_negatives+false_positives+false_negatives)
	
	return [error, recall, precision, truenegrate, accuracy]

def venue_inference(dataset, classifier, venue_type, n_folds=10):
	ylabels = get_venue_labels(dataset, venue_type)
	print len([i for i in ylabels if i == 1])
	folds = cross_validation.KFold(len(dataset), n_folds=n_folds)

	rmse = 0

	errors = []
	for train, test in folds:
		trainset = [dataset[i] for i in train]
		testset = [dataset[i] for i in test]
		trainlabels = [ylabels[i] for i in train]
		testlabels = [ylabels[i] for i in test]
		errors.append(measure_venue_error(trainset, testset, trainlabels, testlabels, classifier))

	#show_most_informative_features(classifier.vectorizer.vectorizer, classifier.classifier, n=20)
	#show_parameters(classifier.classifier)
	return errors

def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

def show_parameters(clf):
	print clf.get_params()

if __name__ =='__main__':
	errors = {}
	full_data = pickle.load(open(sys.argv[1],'rb'))
	#for folds in [10,20,40,100]:
	#		errors[str(folds)] = first_inference(full_data,models.NaiveRegression(),folds)
	#print errors

	for venues in [['Gym / Fitness Center'],['Vegetarian / Vegan Restaurant', 'Vegetarian'], ['High School'], ['Night Club'], ['University'], ['Art Museum'], ['Rock Club']]:
		print "Venue: ", venues
		for folds in [10]:
			errors = venue_inference(full_data, models.NaiveClassifier(), venues, n_folds= folds)
			print folds
			print np.nanmean(errors,0)