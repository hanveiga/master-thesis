import tweepy
import csv
import sys


import twitter_config as tconf
""" Import your app details """

consumer_key = tconf.consumer_key
consumer_secret = tconf.consumer_secret
access_token = tconf.access_token
access_token_secret = tconf.access_token_secret

class twitter_user(object):
	""" user class with tweets, friends and user info """
	def __init__(self, nick_name):
		self.user = api.get_user(nick_name)

	def get_friends(self):
		self.friends = api.friends(screen_name=self.user.screen_name, count=5000)
		return self.friends

	def get_followers(self):
		self.followers = api.followers(screen_name=self.user.screen_name, count=5000)
		return self.followers
		
	def get_tweets(self, output=False):
		# EDIT: Sloppy output, just to check if stuff's working
		count = 0
		self.tweet_list = []
		if output:
			csvFile = open('tweets'+str(self.user.id)+'.csv', 'a')
			csvWriter = csv.writer(csvFile)

		for status in tweepy.Cursor(api.user_timeline, id=self.user.id).items():
			self.tweet_list.append(status)
			if output:
				csvWriter.writerow([str(status.created_at) + ' \t ' + str(status.text.encode('utf-8')) +' \t ' + str(status.geo)])

		return self.tweet_list

	def get_new_tweets(self):
		pass

class tweet(object):
	__slots__ = ('id', 'created_at', 'text', 'geocoords')
	
	def __init__(self, tweet = None, **kwargs):
		if kwargs:
			for attrname in self.__slots__:
				setattr(self, attrname, kwargs.get(attrname,None))
		else:
			self.id = tweet.id
			self.created_at = tweet.created_at
			self.text = tweet.text
			self.geocoords = tweet.geo
			self.tweet = e

if __name__=='__main__':
	""" Tests """
	
	auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
	auth.set_access_token(access_token,access_token_secret)
	api = tweepy.API(auth)

	my_twitter_account = twitter_user(sys.argv[1])
	my_tweets = my_twitter_account.get_tweets(output=True)
	#print my_tweets[0].text
	my_followers = my_twitter_account.get_friends()
	for follower in my_followers:
		print (follower.id, follower.screen_name)