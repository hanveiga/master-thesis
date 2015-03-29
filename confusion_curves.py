# -*- coding: utf-8 -*-
import os
os.environ['MPLCONFIGDIR'] = "/local/.config/matplotlib" #Nasty fix
from sklearn import cross_validation
import cPickle as pickle
import sys
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

import models
from data import *
from find_venue_heuristics import get_top_venues, get_venue_type_visited

from models import get_visited_venue_labels, ProgressiveClassifier, ProgressiveEnsembleTweetClassifier

def oversampler(labels, threshold = 0.35, n_folds=3):
	# Script that balances the classes by oversampling the minority.
	n = len(labels)

	n_class_major = len([i for i in labels if i == 1])
	n_class_minor = n - n_class_major
	minority_label = 0

	if n_class_minor > n_class_major:
		n_class_major, n_class_minor = n_class_minor, n_class_major
		minority_label = 1

	ratio_min_total = n_class_minor/ float(n) 

	indices = range(len(labels))

	if ratio_min_total < threshold:
		to_add = int( (threshold*n - n_class_minor)/float(1-threshold))
		minority_idx = [indx for indx, i in zip(indices,labels) if i == minority_label]
		resampling = np.random.choice( minority_idx, to_add, replace = True)
		extended_indx = list(indices) + list(resampling)
		print 'ratio before oversampling: %s' %(len(minority_idx)/float(len(labels)))
		print 'ratio after oversampling: %s' %((len( minority_idx) + len(resampling)) /float(len(list(labels) + list(resampling))))
		return extended_indx
	else:
		return indices

def train_classifiers(dataset, classifier, list_of_venues):
	# list_of_venues is a list of tuples
	dict_of_classifiers = {}

	for venue in list_of_venues:
		# train classifier
		ylabels = get_visited_venue_labels(dataset, venue)

		extended_indx = oversampler(ylabels, threshold=0.45)
		dataset_extended = [dataset[i] for i in extended_indx]
		y_labels_extended = get_visited_venue_labels(dataset_extended, venue)
		classifier_ven = classifier()
		classifier_ven.train(dataset_extended,y_labels_extended)
		dict_of_classifiers[venue] = classifier_ven

	print dict_of_classifiers
	return dict_of_classifiers

def get_real_labels(user, venue_types):
	labels_dict={}
	all_visited_venues = get_venue_type_visited(user)

	for ven_type in venue_types:
		if ven_type in all_visited_venues:
			labels_dict[ven_type] = 1
		else:
			labels_dict[ven_type] = 0 
	return labels_dict

def IncrementalLearningTweetsMeasure(user, dict_of_classifiers, initial_num_tweets=1):
	# Error per venue type
	random.shuffle(user.twitter)
	truncated_tweets = [ tweet for tweet in user.twitter[:initial_num_tweets]]
	venue_types = dict_of_classifiers.keys()
	real_label = get_real_labels(user,venue_types)

	user_error = []
	incremental_error = []
	confusion_dict = defaultdict(list)

	for key, classifier in dict_of_classifiers.items():
			prediction = classifier.predict([truncated_tweets])
			confusion_dict[key].append(return_confusion(prediction, real_label[key]))

	# Incremental prediction
	increment = 0
	for tweet in user.twitter[initial_num_tweets:]:
			truncated_tweets.append(tweet)
			increment += 1
			if increment % 50 == 0:
			  for key, classifier in dict_of_classifiers.items():
				  prediction = classifier.predict([truncated_tweets])
		          confusion_dict[key].append(return_confusion(prediction, real_label[key]))
			  increment = 0
			else:
			  pass

	# add remaining tweets
	last_tweet = len(truncated_tweets)
	for tweet in user.twitter[last_tweet:]:
		truncated_tweets.append(tweet)

	for key, classifier in dict_of_classifiers.items():
		prediction = classifier.predict([truncated_tweets])
		confusion_dict[key].append(return_confusion(prediction, real_label[key]))

	return confusion_dict

def return_confusion(prediction,real_label):
	if (prediction == real_label) and prediction == 1:
		## true positive
		return 1
	elif (prediction == real_label) and prediction == 0:
		## true negative
		return 2
	elif (prediction != real_label) and prediction == 1:
		## false positive
		return 3
	elif (prediction != real_label) and prediction == 0:
		## false negative
		return 4
	else:
		return 'null'

def get_error_incremental_learning(train, test, classifier_type, list_of_venues):
	# pass a matrix back, users x incrementals
	list_of_classifiers = train_classifiers(train, classifier_type, list_of_venues)
	errors = []
	iterations = 10

	for iteration in range(iterations):
		for user in test:
			error = IncrementalLearningTweetsMeasure(user,list_of_classifiers)
			errors.append(error)
	return errors

def get_errors(dataset, classifier_type, list_of_venues, folds=10):
	folds = cross_validation.KFold(len(dataset),n_folds=folds)	
	errors = []
	count = 0
	for train, test in folds:
		trainset = [dataset[i] for i in train]
		testset = [dataset[i] for i in test]
		error_dictionaries = get_error_incremental_learning(trainset, testset, classifier_type, list_of_venues)
		for error in error_dictionaries:
			errors.append(error)
		count = count+1
		if count > 1:
			break
	return errors

def plot_errors(list_dictionaries, venues):
	num_users = len(list_dictionaries)
	venues = list_dictionaries[0].keys()
	for venue in venues:
		max_len = 0
		error_per_venue = []
		for dictionary in list_dictionaries:
			error_per_venue.append(dictionary[venue])
			if len(dictionary[venue]) > max_len:
				max_len = len(dictionary[venue])
		error_matrix = np.zeros((num_users,max_len))
		error_matrix[:] = np.nan
		for i in range(num_users):
			for j in range(len(error_per_venue[i])):
				error_matrix[i,j] = error_per_venue[i][j]
			if range(len(error_per_venue[i])) < max_len:
				for k in xrange(len(error_per_venue[i]),max_len):
					error_matrix[i,k] = error_per_venue[i][-1]
		plt.title(venue)
		print error_matrix.shape
		plt.plot(np.mean(error_matrix,0))
		plt.show()
		plt.savefig(venue+'.png')
		plt.clf()

	print ' What the fuck '

def plot_confusion(list_dictionaries):
	num_users = len(list_dictionaries)
	venues = list_dictionaries[0].keys()

	print list_dictionaries[0]['Church']

	count = 0
	for venue in venues:
		max_len = 0
		error_per_venue = []
		for dictionary in list_dictionaries:
			error_per_venue.append(dictionary[venue])
			if len(dictionary[venue]) > max_len:
				max_len = len(dictionary[venue])
		error_matrix = np.zeros((num_users,max_len))
		error_matrix[:] = np.nan
		for i in range(num_users):
			for j in range(len(error_per_venue[i])):
				error_matrix[i,j] = error_per_venue[i][j]
			if range(len(error_per_venue[i])) < max_len:
				for k in xrange(len(error_per_venue[i]),max_len):
					error_matrix[i,k] = error_per_venue[i][-1]
		
		#plt.title(venue)
		#print error_matrix.shape
		#plt.plot(np.mean(error_matrix,0))
		#plt.show()
		#plt.savefig(venue+'.png')
		#plt.clf()

		# compute recall, precision, accuracy, true neg rate
		recall_curve = []
		precision_curve = []
		f1_curve = []
		accuracy_curve = []
		true_neg_curve = []
		for k in range(error_matrix.shape[1]): # number of iterations
			iteration = error_matrix[:,k]
			recall = len([a for a in iteration if a == 1])/float(len([a for a in iteration if a == 1 or a == 4]))
			precision = len([a for a in iteration if a == 1])/float(len([a for a in iteration if a == 1 or a == 3]))
			true_neg = len([a for a in iteration if a == 2])/float(len([a for a in iteration if a == 2 or a == 3]))
			recall_curve.append(recall)
			precision_curve.append(precision)
			f1_curve.append(2*precision*recall/float(precision+recall))
			accuracy = len([a for a in iteration if a == 1 or a == 2])/float(len([a for a in iteration]))
			accuracy_curve.append(accuracy)
			true_neg_curve.append(true_neg)

		range_x = len(accuracy_curve)

		plt.title(venue+' Accuracy')
		plt.plot(accuracy_curve, label="Accuracy", marker='.')
		plt.plot(recall_curve, label="Recall", marker='.')
		plt.plot(precision_curve, label="Precision", marker='.')
		plt.plot(f1_curve, label="F1-Score", marker='.')
		plt.plot(true_neg_curve, label="F1-Score", marker='.')
		plt.legend()
		plt.savefig('fig_' +str(count) +'_accuracy'+'.png')
		plt.clf()

		plt.title(venue+' Recall')
		plt.plot(recall_curve)
		plt.savefig('fig_' +str(count) +'_recall'+'.png')
		plt.clf()

		plt.title(venue+' Precision')
		plt.plot(precision_curve)
		plt.savefig('fig_' +str(count) +'_precision'+'.png')
		plt.clf()

		plt.title(venue+' F1 score')
		plt.plot(f1_curve)
		plt.savefig('fig_' +str(count) +'_f1score'+'.png')
		plt.clf()

		count = count + 1




if __name__ =='__main__':
	full_data = pickle.load(open(sys.argv[1],'rb'))

	full_data_2 = []
	for user in full_data:
		if len(user.twitter) >= 2500:
			full_data_2.append(user)
		else:
			pass
	print len(full_data_2)

	#venue_counts = get_top_venues(full_data_2)

	# generate the venue classifier solely based on the frequency of visits
	#severalvenues = venue_counts.most_common()[70:80]
	#list_of_venues=[]
	#for key, value in severalvenues:
	#	list_of_venues.append(key)

	#list_of_venues = ['Gym' , 'Church'] # ,'Wine Bar', 'Gym / Fitness Center', 'Concert Hall', 'Theater', 'Resort', 'Museum', 'Performing Arts Venue', 'College & University', 'Vegetarian / Vegan Restaurant']
	list_of_venues = ['Gym' , 'Wine Bar', 'Theater'] # 'Resort', 'Museum', 'Performing Arts Venue', 'College & University', 'Vegetarian / Vegan Restaurant']


	errors = get_errors(full_data_2, ProgressiveEnsembleTweetClassifier, list_of_venues, folds=10)
	pickle.dump(errors,open('debugging.pkl','wb'))
	
	"""
	#errors = get_errors(full_data_2, ProgressiveEnsembleTweetClassifier, list_of_venues, folds=10)
	list_of_venues = ['Gym' , 'Church'] # ,'Wine Bar', 'Gym / Fitness Center', 'Concert Hall', 'Theater', 'Resort', 'Museum', 'Performing Arts Venue', 'College & University', 'Vegetarian / Vegan Restaurant']


	errors = pickle.load(open('error_matrix_tfidf_confusion_27march.pkl','rb'))

	for venue in list_of_venues:
		plot_confusion(errors)
	"""