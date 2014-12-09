from django.db import models
from django.db.utils import IntegrityError
from django.conf import settings
import logging
from django.http import HttpResponse, HttpResponseRedirect

from models import Twitter_User, Tweets
import re
import twitter_scrapper as crawl_tt

def get_twitter(user, twitter_name):
	print "entered twitter"

	# check if user exists
	if (Twitter_User.objects.filter(username = twitter_name).exists()):
		print "exists already"
		# increment the tweets that haven't been cached
		return HttpResponseRedirect('http://yahoo.com') 

	else:
		crawl_user = crawl_tt.twitter_user(twitter_name)

		twitter_user = Twitter_User(Id = user, username = twitter_name,
				twitter_id = crawl_user.user.id , location = crawl_user.user.location,
				description = crawl_user.user.description,
				followers = str(crawl_user.get_followers()),
				friends = str(crawl_user.get_friends())
				)
		twitter_user.save()
		print "saved twitter user"
		tweets = crawl_user.get_tweets()
		for tweet in tweets:
			add_tweet = Tweets(Id = user, created_at = tweet.created_at,
							text= tweet.text, coords = 'llala',
							hashtags = tweet.hashtags(),links=tweet.links())
			add_tweet.save()
		return ''	