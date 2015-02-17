import urllib2
import sys
from bs4 import BeautifulSoup
import re
import foursquare
import foursqid

# Construct the client object
client = foursquare.Foursquare(client_id=foursqid.client_id, client_secret=foursqid.client_secret)

# Build the authorization url for your app
auth_uri = client.oauth.auth_url()

def get_venue_from_checkin(checkin_link):
	response = urllib2.urlopen(checkin_link)
	soup = BeautifulSoup(response.read())
	url_place = soup.find("meta",{"property":"getswarm:place"})['content']
	venue_id = re.split('/',url_place)[-1]
	return [url_place, venue_id]
		
def get_venue(venue_id):
	return client.venues(venue_id)

if __name__=='__main__':
	url, ven_id = get_venue_from_checkin(sys.argv[1])
	ven = get_venue('49e4a677f964a52014631fe3')	