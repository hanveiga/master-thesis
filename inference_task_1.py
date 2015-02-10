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

	error =  sum([np.abs(x-y) for x,y in zip(predictions,test_labels)])

	print len(test_labels)
	print error 
	return error 	

def first_inference(dataset_name, classifier):
	# load data
	full_data = pickle.load(open(dataset_name,'rb'))

	# exclude users with no price-tag
	temp_data = []
	for user in full_data:
		if not filter(None,user.get_spendings()):
			continue
		else:
			temp_data.append(user)

	full_data = temp_data

	folds = cross_validation.KFold(len(full_data),n_folds=10)

	rmse = 0

	for train, test in folds:
		trainset = [full_data[i] for i in train]
		testset = [full_data[i] for i in test]
		rmse += measure_error(trainset, testset, classifier)

	return rmse / float(len(folds))

if __name__ =='__main__':
	error = first_inference(sys.argv[1],models.NaiveRegression())
	print error