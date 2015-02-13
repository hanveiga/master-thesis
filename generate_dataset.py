# -*- coding: utf-8 -*-
import csv
import users
import sys

def load_users(filepath,csv_file):
	with open(csv_file, 'rb') as csvfile:
		read_users = csv.reader(csvfile, delimiter=';')
		list_of_users = []
		for row in read_users:
			user_dict = {'twitter': row[0], 'instagram': row[1], 'foursquare': row[0]}
			list_of_users.append(user_dict)	
	return list_of_users

def generate_users(list_of_users):
	users.populate_users(filepath,list_of_users)
	print "done"

def stats_about_data(csv_file):
	with open(csv_file, 'rb') as csvfile:
		read_users = csv.reader(csvfile, delimiter=';')
		instagram_count = 0
		foursquare_count = 0
		useful_pair = 0
		total_records = 0
		list_of_users = []
		for row in read_users:
			if len(row) == 0 or len(row) > 4:
				continue
			else:
				total_records += 1
				if row[1] != ['null', 'n']:
					instagram_count += 1
				if row[2] != ['null','n']:
					foursquare_count += 1
					if (row[1] not in ['null','n']):
						useful_pair += 1
						user_dict = {'twitter': row[0], 'instagram': row[1], 'foursquare': row[0]}
						list_of_users.append(user_dict)

	print '------------------------'
	print 'Users scanned: ', total_records
	print 'Users in instagram: ', instagram_count
	print 'Users in foursquare: ', foursquare_count
	print 'Triplets available: ', useful_pair

	return list_of_users

if __name__=='__main__':
	#dict_of_users = load_users('users.csv')
	#print dict_of_users
	#print dict_of_users[0]
	#users.populate_users(dict_of_users)
	dict_of_users = stats_about_data(sys.argv[2])
	print dict_of_users
	users.populate_users(sys.argv[1],dict_of_users)