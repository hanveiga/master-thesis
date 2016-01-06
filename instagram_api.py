# -*- coding: utf-8 -*-
import urllib2
import oembed
from instagram.client import InstagramAPI
import sys
import re

def get_insta_user(short_link, debug=1):
	""" Get instagram userid from a link posted on twitter """
	print " Fetching instagram user "
	try:
		print short_link
		response = urllib2.urlopen(short_link) # Some shortened url, eg: http://t.co/z8P2xNzT8k
		#print response.__dict__
		url_destination_https = response.url
		url_destination = url_destination_https.replace('https','http',1)

		# from link get the media_id
		consumer = oembed.OEmbedConsumer()
		endpoint = oembed.OEmbedEndpoint('https://api.instagram.com/oembed?url=', ['http://instagr.am/p/*'])
		consumer.addEndpoint(endpoint)
		print url_destination
		media_id_code = re.split('/',url_destination)[-2]
		print media_id_code
		url_destination = 'http://instagr.am/p/'+media_id_code
		response = consumer.embed(url_destination)
		print response
		media_id = response['media_id']
		print 'media id'
		print media_id
		print response
		#https://api.instagram.com/oembed/?url=http://instagram.com/p/bNd86MSFv6/&beta=true

		api = InstagramAPI(client_id='d09ed38ad8f14bd7bc7fdf6e7ebd9340', client_secret='0de686fbad2543968159c573f33e9409')
	except:
			if debug:
				print 'Unable to find picture from link.'
			return 'null'

	try:
		media = api.media(media_id)
		#if debug:
		print media.user.id
		return [media.user.id, media.user.username]
	except:
		if debug:
			print 'Unable to fetch instagram ID - most likely private user'
		return 'null'

if __name__=='__main__':
	user = get_insta_user(sys.argv[1])
	print user
