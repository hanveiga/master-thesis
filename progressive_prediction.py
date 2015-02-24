# -*- coding: utf-8 -*-
from sklearn import cross_validation
import cPickle as pickle
import sys
import numpy as np

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

def train_classifiers(dataset, classifier):
	# Train dataset or full dataset (?)

	top_venues = get_top_venues(dataset)

	#list_of_classifiers = []
	dict_of_classifiers = {}

	for key, value in top_venues.most_common(30):
		# train classifier
		#list_of_classifiers.append(classifier)
		ylabels = get_visited_venue_labels(dataset, key)
		#print ylabels
		classifier_ven = classifier
		classifier_ven.train(dataset,ylabels)
		dict_of_classifiers[key] = classifier_ven

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


def IncrementalLearning(user, dict_of_classifiers, initial_num_hash=500):
	user_hashtags = user.get_all_hashtags()

	truncated_hashtags = user_hashtags[:initial_num_hash]

	list_of_testvectors = []

	venue_types = dict_of_classifiers.keys()

	real_label = get_real_labels(user,venue_types)

	user_error = []
	incremental_error = []
	# First prediction

	#print truncated_hashtags

	#try:
	print real_label
	print truncated_hashtags[0:10]
	for key, classifier in dict_of_classifiers.items():
			#transformed = classifier.predict(truncated_hashtags)
			#list_of_testvectors.append(transformed)
			prediction = classifier.predict([truncated_hashtags])
			print prediction
			user_error.append(np.abs(prediction - real_label[key]))

	error = np.sum(user_error)
	print 'Computed first vector'
	incremental_error.append(error)
	#except:
	#	print 'Did not compute first vector'

	try:
		
		# Incremental prediction
		reached_hundred = 0
		for hashtag in user_hashtags[initial_num_hash:]:
			#print hashtag
			truncated_hashtags.append(hashtag)
			user_error = []
			reached_hundred += 1
			if reached_hundred % 50 == 0:
			  for key, classifier in dict_of_classifiers.items():
				  #transformed = classifier.fit(truncated_hashtags)
				  #list_of_testvectors.append(transformed)
				  prediction = classifier.predict([truncated_hashtags])
				  print prediction
				  user_error.append(np.abs(prediction - real_label[key]))
			  error = np.sum(user_error)
			  #print user_error
			  incremental_error.append(error)
			  reached_hundred = 0
			else:
			  pass
		print 'computed the incremental vector'
	except:
		print 'Did not compute the rest of the vectors'

	return incremental_error

def get_error_incremental_learning(train, test, classifier_type):
	# pass a matrix back, users x incrementals

	list_of_classifiers = train_classifiers(train, classifier_type)

	errors = []
	for user in test:
		error = IncrementalLearning(user,list_of_classifiers,initial_num_hash=500)
		errors.append(error)

	max_length = 0
	user_num = len(errors)
	for error in errors:
		if len(error) > max_length:
			max_length = len(error)

	error_matrix = np.empty((max_length,user_num))
	error_matrix[:] = np.nan

#	i = 0
#	j = 0
#	for error in errors:
#		print error_matrix.shape
#		print error
#		for elem in error:
#			print elem
#			error_matrix[j,i] = elem
#			j = j + 1
#		i = i + 1

	# populate error_matrix
	for j in range(user_num):
		for i in range(len(errors[j])):
			error_matrix[i,j] = errors[j][i]


	return error_matrix

def run_crossvalidation(dataset, classifier_type, folds=10):
	folds = cross_validation.KFold(len(dataset),n_folds=40)	

	errors = []

	for train, test in folds:
		trainset = [dataset[i] for i in train]
		testset = [dataset[i] for i in test]
		error = get_error_incremental_learning(trainset, testset, classifier_type)
		errors.append(error)

	# make the error matrix...
	return errors

if __name__ =='__main__':
	full_data = pickle.load(open(sys.argv[1],'rb'))

	full_data_2 = []
	for user in full_data:
		if len(user.twitter) >= 1000:
			full_data_2.append(user)
		else:
			pass

	#list_class = train_classifiers(full_data,ProgressiveClassifier())

	#for key, val in list_class.items():
	#	print key
	#	print val
	#	print val.__dict__
	errors = run_crossvalidation(full_data_2, ProgressiveClassifier(), folds=1)
	print errors
	pickle.dump(errors,open('error_matrix_1.pkl','wb'))