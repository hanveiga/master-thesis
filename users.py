# -*- coding: utf-8 -*-
import tweepy
import twitter_config as tconf
import re
import sys
import time
import datetime
import random
import os.path

import cPickle as pickle

### TWITTER API
consumer_key = tconf.consumer_key
consumer_secret = tconf.consumer_secret
access_token = tconf.access_token
access_token_secret = tconf.access_token_secret

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth)

### INSTAGRAM API
from instagram.client import InstagramAPI

api_instagram = InstagramAPI(client_id='d09ed38ad8f14bd7bc7fdf6e7ebd9340', client_secret='0de686fbad2543968159c573f33e9409')

def check_remaining_calls(api_tweet):
	try:
		remaining = int(api_tweet.last_response.headers['x-rate-limit-remaining'])
		#print remaining
		return remaining
	except:
		print 'There is no last response yet'
		return 'null'
	#remaining = api_tweet.rate_limit_status()

def remaining_response(remaining):
	if remaining <= 2:
		print "entered sleep period"
		time.sleep(15*60 + 10)
		print "exited sleep period"
		U = api.get_user(screen_name='hanveiga') #just to reset the thing
		#print U
	else:
		pass

class User(object):
	def __init__(self, twitter_username, instagram_username, foursquare_username):
		self.twitter = Twitter_Profile(twitter_username)
		self.instagram = Instagram_Profile(instagram_username)
		self.foursquare = Foursquare_Profile(foursquare_username) 

		#return 'null'

def update_check_ins(user):
		""" Add check ins from the timeline of the guy, instead of relying on the twitter search api """
		list_of_check_in_ids=[]
		for check_in in user.foursquare.check_ins:
			list_of_check_in_ids.append(check_in.id)

		for tweet in user.twitter.timeline:
			regex = r'https?://swarmapp\.com+|www\.swarmapp\.com+'
			#print tweet.entities['urls']
			if tweet.entities['urls']: #if non-empty, find the url.
				urls = tweet.entities['urls'][0]
				#print urls['expanded_url']
				match = re.search(regex, urls['expanded_url'])   	
				if match:
						#print urls['expanded_url']
						if tweet.id in list_of_check_in_ids:
							continue
						else: 
							user.foursquare.check_ins.append(tweet)
			else:
				continue

class Twitter_Profile(object):
	def __init__(self,twitter_username):
		self.__username = twitter_username
		self.profile = self.get_userprofile()
		self.timeline = self.get_timeline()

	def get_userprofile(self):
		r = check_remaining_calls(api)
		remaining_response(r)
		print "got twitter profile"
		return api.get_user(screen_name=self.__username)

	def get_timeline(self):
		""" return list of tweets """
		r = check_remaining_calls(api)
		remaining_response(r)
		timeline=[]
		c = tweepy.Cursor(api.user_timeline, id=self.__username, include_entities=True)
		r = check_remaining_calls(api)
		remaining_response(r)
		non_english = 0
		for tweet in c.items():
			r = check_remaining_calls(api)
			remaining_response(r)
			timeline.append(tweet)
			if len(timeline) == 300:
				# make the check of whether the feed is in english
				#random_indices = random.sample(xrange(len(timeline)), 50)
				print "making the language check"
				for tweet in timeline:
					#print tweet.lang
					if tweet.lang != 'en':
						non_english += 1
						#print "not english"
				print 'non english percentage: ', non_english/float(len(timeline)) 
				if non_english/float(len(timeline)) > 0.2:
					print 'Too little english in the feed'
					#raise Exception("Too little english in the feed")
					break
		print "got twitter timeline"
		return timeline

	def get_last_updates(self):
		return 'todo'

class Instagram_Profile(object):
	def __init__(self, instagram_username):
		#print instagram_username
		#search = api_instagram.user_search(q=instagram_username, count=1)[0]
		#print search.id
		self.__username = instagram_username
		print instagram_username
		self.profile = self.get_userprofile()
		print "got insta user profile"
		self.timeline = self.get_timeline()
		print "got insta timeline"

	def get_userprofile(self):
		return api_instagram.user(user_id=self.__username)

	def get_timeline(self):
		return api_instagram.user_recent_media(user_id=self.__username,count=1000)

	def get_last_updates(self):
		return 'todo'

class Foursquare_Profile(object):
	""" Foursquare_Profile is generated through the twitter timeline, so it's not really a foursquare profile """
	def __init__(self, foursquare_username):
		self.__username = foursquare_username
		self.check_ins = self.get_check_ins()

	def get_check_ins(self):
		r = check_remaining_calls(api)
		remaining_response(r)
		c = tweepy.Cursor(api.search, q='swarmapp.com'+ ' -RT' + ' from:'+self.__username, lang="en")
		check_ins = []
		r = check_remaining_calls(api)
		remaining_response(r)

		for tweet in c.items():
			#print tweet.text.encode('cp850', errors='replace').decode('cp850')
			r = check_remaining_calls(api)
			remaining_response(r)
			check_ins.append(tweet)
		print "got foursquare checkins"
		return check_ins

		# search through the timeline of the guy instead 

class Foursquare_entry(object):
	def __init__(self):
		self.link = 'a'
		self.tweet = 'b'

	def get_category(self):
		category = 'null'
		return category

def generate_pickle(filepath,user_nicknames):
	#""" Receives a dictionary with the nicknames """

	try: 

		user = User(twitter_username=user_nicknames['twitter'],
				instagram_username=user_nicknames['instagram'],
				foursquare_username=user_nicknames['foursquare'])
		update_check_ins(user)

		pickle.dump(user,open(filepath+user_nicknames['twitter']+'.pkl','wb'))	
		return user
	except:
		print "Unable to fetch user"
		user = {}
		pickle.dump(user,open(filepath+user_nicknames['twitter']+'.pkl','wb'))
		return user

def populate_users(filepath, userdict_list):
	# generates a list of users
	users = []
	for user_nicknames in userdict_list:
		if os.path.isfile(filepath+user_nicknames['twitter']+'.pkl'):
			print 'user exists'
			continue
		a = generate_pickle(filepath, user_nicknames)

if __name__=='__main__':
	userdict_list = []
	userA = {}
	userA['twitter'] = 'hanveiga'
	userA['instagram'] = 'whoisthis2'
	userA['foursquare'] = 'hanveiga'
	userdict_list.append(userA)
	populate_users(sys.argv[1], userdict_list)