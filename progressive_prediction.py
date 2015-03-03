# -*- coding: utf-8 -*-
from sklearn import cross_validation
import cPickle as pickle
import sys
import numpy as np
import random

import models
from data import *
from find_venue_heuristics import get_top_venues

from models import get_visited_venue_labels, ProgressiveClassifier

# need several classifiers

# TO DO:
# Train several binary classifiers which tell us whether user goes to venue type
# Truncate the hashtag vector
# Make prediction vector (each entry i attains value 0 or 1, whether user goes to venue type i) - record error
# Increment hashtag vector
# Predict and record error
# Repeat this until we recover the full hashtag vector
# Do this accross all test examples
# Plot individual curves and average curves

def train_classifiers(dataset, classifier, list_of_venues):
	# list_of_venues is a list of tuples
	dict_of_classifiers = {}

	for venue, val in list_of_venues:
		# train classifier
		ylabels = get_visited_venue_labels(dataset, venue)
		classifier_ven = classifier
		classifier_ven.train(dataset,ylabels)
		dict_of_classifiers[venue] = classifier_ven

	print dict_of_classifiers
	return dict_of_classifiers

def get_real_labels(user, venue_types):

	labels_dict={}
	all_visited_venues = []

	for checkin in user.foursquare:
		for category in checkin.lowest_type:
			if category in all_visited_venues:
				pass
			else:
				all_visited_venues.append(category)

	for ven_type in venue_types:
		if ven_type in all_visited_venues:
			labels_dict[ven_type] = 1
		else:
			labels_dict[ven_type] = 0 

	return labels_dict


def IncrementalLearning(user, dict_of_classifiers, initial_num_hash=100):
	user_hashtags = user.get_all_hashtags()

	random.shuffle(user_hashtags)

	truncated_hashtags = user_hashtags[:initial_num_hash]

	list_of_testvectors = []

	venue_types = dict_of_classifiers.keys()

	real_label = get_real_labels(user,venue_types)

	# First prediction

	#try:
	user_error = []
	incremental_error = []

	print real_label
	print truncated_hashtags[0:10]
	for key, classifier in dict_of_classifiers.items():
			prediction = classifier.predict([truncated_hashtags])
			user_error.append(np.abs(prediction - real_label[key]))

	error = np.sum(user_error)

	incremental_error.append(error)

	try:
		# Incremental prediction
		increment = 0
		for hashtag in user_hashtags[initial_num_hash:]:
			#print hashtag
			truncated_hashtags.append(hashtag)
			user_error = []
			increment += 1
			if increment % 50 == 0:
			  for key, classifier in dict_of_classifiers.items():
				  prediction = classifier.predict([truncated_hashtags])
				  user_error.append(np.abs(prediction - real_label[key]))
			  error = np.sum(user_error)
			  incremental_error.append(error)
			  increment = 0
			else:
			  pass
		print 'computed the incremental vector'
	except:
		print 'Did not compute the rest of the vectors'

	return incremental_error

def get_error_incremental_learning(train, test, classifier_type, list_of_venues):
	# pass a matrix back, users x incrementals

	list_of_classifiers = train_classifiers(train, classifier_type, list_of_venues)

	errors = []
	for user in test:
		error = IncrementalLearning(user,list_of_classifiers,initial_num_hash=10)
		errors.append(error)

	max_length = 0
	user_num = len(errors)
	for error in errors:
		if len(error) > max_length:
			max_length = len(error)

	error_matrix = np.empty((max_length,user_num))
	error_matrix[:] = np.nan

	# populate error_matrix
	for j in range(user_num):
		for i in range(len(errors[j])):
			error_matrix[i,j] = errors[j][i]

	return error_matrix

def run_crossvalidation(dataset, classifier_type, list_of_venues, folds=10):

	folds = cross_validation.KFold(len(dataset),n_folds=folds)	

	errors = []

	for train, test in folds:
		trainset = [dataset[i] for i in train]
		testset = [dataset[i] for i in test]
		error = get_error_incremental_learning(trainset, testset, classifier_type, list_of_venues)
		errors.append(error)

	return errors

if __name__ =='__main__':
	full_data = pickle.load(open(sys.argv[1],'rb'))

	full_data_2 = []
	for user in full_data:
		if len(user.get_all_hashtags()) >= 1200:
			full_data_2.append(user)
		else:
			pass
	print len(full_data_2)

	venue_counts = get_top_venues(full_data_2)
	# generate the venue classifier solely based on the frequency of visits
	list_of_venues = venue_counts.most_common()[60:80]
	print list_of_venues

	errors = run_crossvalidation(full_data_2, ProgressiveClassifier(), list_of_venues, folds=20)
	pickle.dump(errors,open('error_matrix_randomsampled.pkl','wb'))