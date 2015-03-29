import tweepy
import numpy as np
from collections import Counter
import sys
import os
import cPickle as pickle

import plots
import foursquare_checkin as fs
from users import *

#client = foursquare.Foursquare(client_id='BRVPSMEVTSAK30IZZEGGVA4RPAL41AL0NNU15NR3UBMEPDB3', client_secret='5FQLP0MUYRHIPDSZQU1XNZEF4FRXRYCKUCFBCFHDYQUJGWGE')
#auth_uri = client.oauth.auth_url()

#categories_raw = client.venues.categories()

""" Generate plots """
def get_parent(category):
	return parent_category[category]

def get_counter_checkin_venue(user_list):

	checkin_counter = Counter()

	for username in user_list:
		user = pickle.load(open('data/post/'+username, 'rb'))

		for check_in in user.foursquare.check_ins:
			try: 
				#category = (check_in[1]['venue']['categories'][0]['pluralName'])
				categories = [a for a['name'] in check_in[1]['venue']['categories']]
				print categories
				#category_parent = category

				#checkin_counter[category_parent] +=1
				for category in categories:
					checkin_counter[category] += 1

			except:
				pass

	return checkin_counter

def get_counter_users_venue():
	pass

def get_counter_location_users(user_list):

	location_counter = Counter()

	for username in user_list:
		user = pickle.load(open('data/post/'+username, 'rb'))
		location = user.twitter.profile.location.encode('cp850', errors='replace').decode('cp850')
		print location
		#for user in user.twitter.profile:
		try: 
			#location = user.location
			location_counter[location] +=1
		except:
			location_counter['NA'] +=1
	pass

	return location_counter

def get_counter_venue_country(user_list):
	
	country_counter = Counter()

	for username in user_list:
			user = pickle.load(open('data/post/'+username, 'rb'))

			for check_in in user.foursquare.check_ins:
				try: 
					category = (check_in[1]['venue']['location']['country'])
					category_parent = category
					checkin_counter[category_parent] +=1
				except:
					pass

	return country_counter

def get_counter_checkin_city(user_list):

	city_counter = Counter()

	for username in user_list:
			user = pickle.load(open('data/post/'+username, 'rb'))

			for check_in in user.foursquare.check_ins:
				try: 
					city = (check_in[1]['venue']['location']['city'])
					city_counter[city] +=1
				except:
					pass

	return city_counter

def save_text_file(counter, textname):
	text_file = open(textname+".txt", "w")
	for key in counter.keys():
		try:
			text_file.write("%s: %s \n" % (key, counter[key]))
		except:
			text_file.write("Unable to save \n")

	text_file.close()

def get_statistics(user_list):
	statistics = {}
	ven_city_counter = Counter()
	ven_country_counter = Counter()
	ven_type_counter = Counter()
	user_location_counter = Counter()

	for username in user_list:
		user = pickle.load(open('data/post/'+username, 'rb'))

		for check_in in user.foursquare.check_ins:
				try: 
					city = (check_in[1]['venue']['location']['city'])
					ven_city_counter[city] +=1
				except:
					pass

				try: 
					country = (check_in[1]['venue']['location']['country'])
					ven_country_counter[country] +=1
				except:
					pass

				try: 
					#category = (check_in[1]['venue']['categories'][0]['pluralName'])
					#category_parent = category
					#ven_type_counter[category_parent] +=1
					categories = [a['name'] for a in check_in[1]['venue']['categories']]
					print categories
					
					for category in categories:
						ven_type_counter[category] += 1
				except:
					pass

		try:
			location = user.twitter.profile.location.encode('cp850', errors='replace').decode('cp850')
			user_location_counter[location] +=1
		except:
			user_location_counter['NA'] +=1
			pass

		statistics['ven_city'] = ven_city_counter
		statistics['ven_country'] = ven_country_counter
		statistics['ven_type'] = ven_type_counter
		statistics['user_loc'] = user_location_counter

	return statistics

def generate_statistics(statistics_dict):
	for name in statistics_dict.keys():
		save_text_file(statistics_dict[name], name)

def get_facts(statistics_dict):
	check_ins = 0
	for key in statistics_dict['ven_type'].keys():
		check_ins += statistics_dict['ven_type'][key]

	print "Total number of check-ins: %s" %check_ins
	print "Total number of cities: %s" %len(statistics_dict['ven_city'].keys()) 
	print "Total number of countries: %s" %len(statistics_dict['ven_country'].keys()) 
	print "Total number of venue types: %s" % len(statistics_dict['ven_type'].keys())

def run(filepath):
	user_list = os.listdir(filepath)
	statistics = get_statistics(user_list)
	pickle.dump(statistics,open('statistics'+'.pkl','wb'))
	get_facts(statistics)
	#generate_statistics(statistics)

if __name__=='__main__':
	run('data/post/')

	"""user_list = os.listdir('data/post/')
	print user_list
	counts_check_in = get_counter_checkin_venue(user_list)
	counts_user_location = get_counter_location_users(user_list)
	counts_checkin_city = get_counter_checkin_city(user_list)
	save_text_file(counts_check_in, 'check_in')
	save_text_file(counts_user_location, 'user_location')
	save_text_file(counts_checkin_city, 'checkin_city')
	#plots.plot_counter(counts, imagename='locationusers')
	"""