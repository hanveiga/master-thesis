# -*- coding: utf-8 -*-
import random
import numpy as np
import cPickle as pickle
import sys
import os
import matplotlib.pyplot as plt

from models_deconstructed import *
from models import get_visited_venue_labels, ProgressiveClassifier, ProgressiveEnsembleTweetClassifier
import information_measure as im

from sklearn import cross_validation
from collections import defaultdict

import models
from data import *
from find_venue_heuristics import get_top_venues, get_venue_type_visited

def ActiveLearningUser(user, dict_of_classifiers, initial_num_tweets=1):
	#random.shuffle(user.twitter)
	venue_types = dict_of_classifiers.keys()
	real_label = get_real_labels(user,venue_types)

	# Generate sequence of indices
	total_tweets = len(user.twitter)
	ordered_indices = range(total_tweets)
	random_indices = range(total_tweets)
	remaining_indices = range(total_tweets)
	random.shuffle(random_indices)

	user_error = []
	
	error_dict = defaultdict(list)
	information_gain = defaultdict(list)
	accuracy_improv = defaultdict(list)

	vectorization = {}
	vectorization_new = {}
	list_of_vectorizers = {}
	list_of_classifiers_standalone = {}

	# Get standalone vectorizer and classifier
	for key, classifier in dict_of_classifiers.items():
		list_of_vectorizers[key] = LemmatizedStandAloneVectorizer(classifier.vectorizer)
		list_of_classifiers_standalone[key] = ClassifierStandAlone(classifier)

	# get relevancy of all tweets
	relevancy_dict = {}
	vectorized_tweets = {}
	for key, classifier in dict_of_classifiers.items():
		vectorized_tweets[key] = [list_of_vectorizers[key].transform([tweet]) for tweet in user.twitter]
		relevancy_dict[key] = [get_relevancy(list_of_classifiers_standalone[key], tweet) for tweet in vectorized_tweets[key]]

	#print relevancy_dict
	remaining_tweets_vectorization = {}
	for key, vectorized in vectorized_tweets.items():
		remaining_tweets_vectorization[key] = zip(ordered_indices, vectorized)

	truncated_tweets = [ tweet for ind, tweet in zip(ordered_indices,user.twitter) if 
						 ind in random_indices[:initial_num_tweets]]

	added_indices = []
	for ind in random_indices[:initial_num_tweets]:
		#user.twitter.pop(ind)
		remaining_indices.remove(ind)
		added_indices.append(ind)

	for key, _ in remaining_tweets_vectorization.items():
		remaining_tweets_vectorization[key] = list(filter(lambda x: x[0] not in random_indices[:initial_num_tweets], remaining_tweets_vectorization[key] ))

	# first pass
	for key, classifier in dict_of_classifiers.items():
			vectorization[key] = list_of_vectorizers[key].transform(truncated_tweets)
			prediction = list_of_classifiers_standalone[key].predict(vectorization[key].toarray())
			error_dict[key].append(return_confusion(prediction,real_label[key]))
			information_gain[key].append([0,0])
			accuracy_improv[key].append([0,0])

	# given what has been seen before, rank guys according to novelty + relevancy

	skip = 1
	iterations = 0
	while len(remaining_indices) > 1:
			for key, classifier in dict_of_classifiers.items():
				#remaining_tweets_vectorization = [tweet for ind, tweet in zip(ordered_indices, vectorized_tweets[key]) if ind in remaining_indices]
				remaining_tweets_vectorization[key] = list(filter(lambda x: x[0] in remaining_indices, remaining_tweets_vectorization[key] ))
				added_tweets_vectorization = list_of_vectorizers[key].transform(truncated_tweets) 
				novelty_vector = []
				for remaining_tweet_v in remaining_tweets_vectorization[key]:
					#print remaining_tweet_v[1]
					novelty = im.similarity(remaining_tweet_v[1], added_tweets_vectorization)
					novelty_vector.append(novelty)
				#print relevancy_dict[key]
				relevancy_vector = [ relevancy for ind, relevancy in zip(ordered_indices,relevancy_dict[key]) if ind in remaining_indices]
				#print relevancy_vector
				#print novelty_vector
				alpha = 0.7
				information_vector = []
				for nov, rel in zip( novelty_vector, relevancy_vector ):
				#	print alpha * nov + (1-alpha)* rel
					information_vector.append(alpha * nov + (1-alpha)* rel)
				ordered_information = zip(remaining_indices,information_vector)
				ordered_information.sort(key = lambda x:x[1], reverse=True)
				#print ordered_information[0:100]
				#print ordered_information[0]
				#print user.twitter[ordered_information[0][0]].text

				#new_tweet_to_add = []
				for to_add in range(skip):
#					new_tweet_to_add.append(user.twitter[ordered_information[to_add][0]])
					truncated_tweets.append(user.twitter[ordered_information[to_add][0]])
					remaining_indices.remove(ordered_information[to_add][0])
					added_indices.append(ordered_information[to_add][0])
				#print len(remaining_indices)
				#truncated_tweets.append(new_tweet_to_add)

				vectorization = list_of_vectorizers[key].transform(truncated_tweets)

				prediction = list_of_classifiers_standalone[key].predict(vectorization.toarray())
				error_dict[key].append(return_confusion(prediction,real_label[key]))
				iterations += skip
				#print iterations
				#print skip
				if iterations == 50:
					skip = 500
				elif iterations >= len(ordered_indices) - 500:
					print iterations
					print len(ordered_indices)
					print len(remaining_indices)
					#return error_dict
					skip = len(ordered_indices) - iterations - 1

	print error_dict

	return error_dict

def get_relevancy_vector(tweets, classifier):
	relevancy_vector = [get_relevancy(classifier, tweet) for tweet in tweets]
	return relevancy_vector

def get_relevancy(classifier, vector):
	feature_names = classifier.classifier.vectorizer.vectorizer.get_feature_names()
	feature_imp = classifier.classifier.classifier.feature_importances_
	relevancy = 0
	_ , non_empty_cols = vector.nonzero()
	for value in non_empty_cols:
		relevancy += feature_imp[value]
	return relevancy

def get_novelty_vector(tweets, seen_tweets, novelty_measure):
	novelty_vector = [novelty_measure(seen_tweets, tweet) for tweet in tweets]
	return novelty_vector

def information_vector(novelty_vector, relevancy_vector, function):
	information_vector = [get_information(novelty,relevancy, function) for novelty, relevancy
						  in zip(novelty_vector, relevancy_vector)]
	return information_vector

def get_information(novelty, relevancy, function):
	return function(novelty,relevancy)

def linear_information(novelty, relevancy, alpha=0.5):
	return alpha*novelty + (1-alpha) * novelty

def exp_information(novelty, relevancy):
	return np.exp(novelty)*np.exp(relevancy)

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

		extended_indx = oversampler(ylabels, threshold=0.25)
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

def return_confusion(prediction,real_label):
	if ((prediction == real_label) and prediction == 1):
		## true positive
		return 1
	elif ((prediction == real_label) and prediction == 0):
		## true negative
		return 2
	elif ((prediction != real_label) and prediction == 1):
		## false positive
		return 3
	elif ((prediction != real_label) and prediction == 0):
		## false negative
		return 4
	else:
		return 100

def get_error_incremental_learning(train, test, classifier_type, list_of_venues):
	# pass a matrix back, users x incrementals
	list_of_classifiers = train_classifiers(train, classifier_type, list_of_venues)
	errors = []
	iterations = 1
	information_gain = []
	accuracy_gain = []
	for user in test:
		for iteration in range(iterations):
			error = ActiveLearningUser(user,list_of_classifiers)
			#print infgain
			errors.append(error)
			#information_gain.append(infgain)
			#accuracy_gain.append(accgain)
	return errors#, information_gain, accuracy_gain

def get_errors(dataset, classifier_type, list_of_venues, folds=10):
	folds = cross_validation.KFold(len(dataset),n_folds=folds)	
	errors = []
	infos = []
	accs = []
	count = 0
	for train, test in folds:
		trainset = [dataset[i] for i in train]
		testset = [dataset[i] for i in test]
		error_dictionaries = get_error_incremental_learning(trainset, testset, classifier_type, list_of_venues)
		for error in error_dictionaries:
			errors.append(error)
		#for info in inf_dictionaries:
		#	infos.append(info)
		#for acc in acc_dictionaries:
		#	accs.append(acc)
		count = count+1
		if count > 1:
			break
	return errors#, infos, accs


def plot_confusion(list_dictionaries):
	num_users = len(list_dictionaries)
	venues = list_dictionaries[0].keys()

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
			if len(error_per_venue[i]) < error_matrix.shape[1]:
				for k in xrange(len(error_per_venue[i]),max_len):
					error_matrix[i,k] = error_per_venue[i][-1]

		recall_curve = []
		precision_curve = []
		f1_curve = []
		accuracy_curve = []
		true_neg_curve = []
		for k in range(error_matrix.shape[1]): # number of iterations
			iteration = error_matrix[:,k]
			try:
				recall = len([a for a in iteration if a == 1])/float(len([a for a in iteration if a == 1 or a == 4]))
			except:
				recall = 0
			try:
				precision = len([a for a in iteration if a == 1])/float(len([a for a in iteration if a == 1 or a == 3]))
			except:
				precision = 0
			try:	
				true_neg = len([a for a in iteration if a == 2])/float(len([a for a in iteration if a == 2 or a == 3]))
			except:
				true_neg = 0
			recall_curve.append(recall)
			precision_curve.append(precision)
			try:
				f1_curve.append(2*precision*recall/float(precision+recall))
			except:
				f1_curve.append(0)
			accuracy = len([a for a in iteration if a == 1 or a == 2])/float(len([a for a in iteration]))

			accuracy_curve.append(accuracy)
			true_neg_curve.append(true_neg)

		range_x = len(accuracy_curve)

		plt.title(venue+' Accuracy')
		plt.plot(accuracy_curve, label="Accuracy", marker='.')
		plt.plot(recall_curve, label="Recall", marker='.')
		plt.plot(precision_curve, label="Precision", marker='.')
		plt.plot(f1_curve, label="F1-Score", marker='.')
		plt.plot(true_neg_curve, label="True neg rate", marker='.')
		plt.legend(loc=0)

		fig1 = plt.gcf()
		fig1.set_size_inches(18.5,10.5)

		plt.show()
		plt.savefig('1april_similarity_' +str(count) +'_accuracy'+'.png')
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
	
	list_of_venues = ['Church']#, 'Furniture / Home Store', 'Japanese Restaurant', 'Resort', 'Taco Place']
	errors = get_errors(full_data_2, ProgressiveEnsembleTweetClassifier, list_of_venues, folds=10)
	pickle.dump(errors,open('23_april_delta07.pkl','wb'))
	#pickle.dump(infos,open('16april_informationgain_debug2_gym.pkl','wb'))
	#pickle.dump(accs,open('16april_accgain_debug2_gym.pkl','wb'))
	#errors = pickle.load(open('23_april_delta04.pkl','rb'))
	#print errors
	plot_confusion(errors)