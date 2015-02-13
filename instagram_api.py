# -*- coding: utf-8 -*-
import urllib2
import oembed
from instagram.client import InstagramAPI
import sys

def get_insta_user(short_link, debug=0):
	""" Get instagram userid from a link posted on twitter """
	try:
		print short_link
		response = urllib2.urlopen(short_link) # Some shortened url, eg: http://t.co/z8P2xNzT8k
		url_destination = response.url

		# from link get the media_id
		consumer = oembed.OEmbedConsumer()
		endpoint = oembed.OEmbedEndpoint('http://api.instagram.com/oembed', ['http://instagram.com/p/*'])
		consumer.addEndpoint(endpoint)
		response = consumer.embed(url_destination)
		media_id = response['media_id']

		api = InstagramAPI(client_id='d09ed38ad8f14bd7bc7fdf6e7ebd9340', client_secret='0de686fbad2543968159c573f33e9409')
	except:
		if debug:
			print 'Unable to find picture from link.'
		return 'null'

	try:
		media = api.media(media_id)
		if debug:
		  print media.user.id
		return [media.user.id, media.user.username]
	except:
		if debug:
			print 'Unable to fetch instagram ID - most likely private user'
		return 'null'

if __name__=='__main__':
	user = get_insta_user(sys.argv[1])
	print user