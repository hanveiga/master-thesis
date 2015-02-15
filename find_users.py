# -*- coding: utf-8 -*-
import tweepy
import re
import sys
from collections import defaultdict
import cPickle as pickle
import csv
import time

import twitter_config as tconf
from instagram_api import get_insta_user
import check_remaining_calls as limit

consumer_key = tconf.consumer_key
consumer_secret = tconf.consumer_secret
access_token = tconf.access_token
access_token_secret = tconf.access_token_secret

def search_on_user(api, user_name, search_term):
    """ Searches a term over a user's twitter feed """
    limit.check_remaining_calls(api)
    c = tweepy.Cursor(api.search, q=search_term+ ' -RT' + ' from:'+user_name, lang="en") # Removes retweets
    limit.check_remaining_calls(api)
    list_of_tweets = []
    counter = 0
    for tweet in c.items():
        limit.check_remaining_calls(api)
        counter = counter + 1
        tweet_text = tweet.text.encode('cp850', errors='replace').decode('cp850')
        regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        match = re.search(regex, tweet_text)
        if match:
            link = match.group()
            list_of_tweets.append(link)
        #if counter > 0:
        #    break # this essentially makes the loop only look at the first result from the search... makes code quicker but may be prone to errors
    if counter == 0:
        return 'null' 
    
    return list_of_tweets[0]

def search_tweets_by_key(auth, xpost_term, xpost_term_2,filename):
    api = tweepy.API(auth)

    limit.check_remaining_calls(api)
    c = tweepy.Cursor(api.search, q=xpost_term + ' -RT' + ' lang:en', lang="en")
    limit.check_remaining_calls(api)
    
    csv_file = open(filename+'.csv','a')
    csv_writer = csv.writer(csv_file,delimiter=';')
    
    linking = []

    for tweet in c.items():
        limit.check_remaining_calls(api)
        tweet_text = tweet.text.encode('cp850', errors='replace').decode('cp850')
        regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        match = re.search(regex, tweet_text)
        if match:
            link = match.group()
	    limit.check_remaining_calls(api)
            response = search_on_user(api,tweet.user.screen_name,xpost_term_2)
            limit.check_remaining_calls(api)
	    #print response
            if response is 'null':
                try:
                    user = [tweet.user.screen_name,response,link]
                    csv_writer.writerow(user)
                    print user
                except:
                    pass
            else:
                try:
                    print 'found instagram'
                    instagram_user = get_insta_user(response)
                    if len(instagram_user[0]) < 2:
                        print "name too short, probably error"
                        continue
                    user = [tweet.user.screen_name,instagram_user[0],link]
                    print [tweet.user.screen_name,instagram_user[1],link]
                    linking.append(user)
                    csv_writer.writerow(user)
                except:
                    pass

        if len(linking)>=500: # just some limit of found users, used for testing code
            break

    csv_file.close()    


if __name__=='__main__':
    auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_token,access_token_secret)
    search_tweets_by_key(auth, 'swarmapp.com','instagram.com','users_file_foursquare')
    #test()
