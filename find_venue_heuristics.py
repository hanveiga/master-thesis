import cPickle as pickle
import sys
from collections import Counter

from data import *

def get_venue_type_visited(user):

	list_venue_types = []

	for checkin in user.foursquare:
		for category in checkin.lowest_type:
			if category not in list_venue_types:
				list_venue_types.append(category)
			else:
				pass

	return list_venue_types

def get_top_venues(dataset, n=20):
	""" Find venue types which are well represented in our dataset
	This means, enough users checked in (but not too many have checked in)
	eg: 30 out of 100 users have visited these venues """

	category_counter = Counter()

	dataset_size = len(dataset)

	for user in dataset:
		list_visited_venues = get_venue_type_visited(user)
		for venuetype in list_visited_venues:
			category_counter[venuetype] += 1

	# get the percentage of users visiting the venues
	
	total = sum(category_counter.values())

	for key in category_counter.keys():
		category_counter[key] = category_counter[key]/float(dataset_size)

	return category_counter

if __name__ == '__main__':
	dataset = pickle.load(open(sys.argv[1],'rb'))
	breakdown = top_venues(dataset)
	for key, val in breakdown:
		print '| %s | %s |' %(key, val*100)