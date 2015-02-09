import cPickle as pickle
import os

class User(object):
	def __init__(self, unpickled):
		self.name = unpickled.twitter.profile.screen_name
		self.foursquare = self.__get_checkins(unpickled)
		self.twitter = self.__get_tweets(unpickled)
		self.instagram = self.__get_instaphotos(unpickled)

	def __get_objects(self, unpickled_timeline, constructor):
		timeline = []

		for raw_object in unpickled_timeline:
			new_object = constructor(raw_object)
			timeline.append(new_object)

		return timeline

	def __get_tweets(self, unpickled):
		tweet_timeline = []
		for raw_tweet in unpickled.twitter.timeline:
			tweet = Tweet(raw_tweet)
			tweet_timeline.append(tweet)

		return tweet_timeline

	def __get_checkins(self, unpickled):
		foursquare_timeline = []

		for raw_checkin in unpickled.foursquare.check_ins:
			if isinstance(raw_checkin[1], basestring):
				continue
			checkin = Checkin(raw_checkin)
			foursquare_timeline.append(checkin)

		return foursquare_timeline

	def __get_instaphotos(self, unpickled):
		insta_timeline = []

		#for raw_photo in unpickled.instagram.timeline:
		#	photo = Tweet(raw_photo)
		#	insta_timeline.append(photo)

		return insta_timeline

	def most_visited_places(self):
		pass

	def most_visited_main_types(self):
		pass

	def most_visited_type(self):
		pass

	def most_common_hashtags(self, n=10):
		pass

	def get_all_hashtags(self):
		return [hashtag for tweet in self.twitter for hashtag in tweet.hashtags]

	def get_spendings(self):
		return [checkin.price for checkin in self.foursquare]

	def get_visited_places(self):
		pass

class Tweet(object):
	def __init__(self, tweet_raw):
		self.text = tweet_raw.text.encode('cp850', errors='replace').decode('cp850')
		self.created_at = tweet_raw.created_at
		self.hashtags = [hashtag['text'].encode('cp850', errors='replace').decode('cp850')
							for hashtag in tweet_raw.entities['hashtags']]
		self.url = [url['expanded_url'] for url in tweet_raw.entities['urls']]
		self.geo = tweet_raw.coordinates
		self.lang = tweet_raw.lang

class Checkin(object):
	def __init__(self, checkin):
		self.venue_name = checkin[1]['venue']['name']
		self.main_type = [category['name'] for category in checkin[1]['venue']['categories']]
		self.sub_type = [category['name'] for category in checkin[1]['venue']['categories']]
		self.lowest_type = [category['name'] for category in checkin[1]['venue']['categories']]
		self.tags = checkin[1]['venue']['tags']
		self.created_at = checkin[0].created_at
		try:
			self.city = checkin[1]['venue']['location']['city']
		except:
			self.city = 'unknown'

		try:
			self.country = checkin[1]['venue']['location']['country']
		except:
			self.country = 'unknown'

		self.url = checkin[1]['venue']['canonicalUrl']
		self.tweet = Tweet(checkin[0])
		try:
			self.price = checkin[1]['venue']['price']['tier']
		except:
			self.price = None

	def get_main_type_name(self):
		return mapping[self.main_type]

	def get_category_names(self):
		return [mapping(mapping_lowest_to_main(category))
							 for category in self.categories]

class Instaphoto(object):
	def __init__(self, instaphoto_raw):
		return instaphoto_raw
#		self.
#		self.text
#		self.hashtags
#		self.geo

def load_raw_users(filepath, userlist):
	users = []

	for username in userlist:
		user_unpickled = pickle.load(open(filepath+username,'rb'))
		user = User(user_unpickled)
		users.append(user)

	pickle.dump(users,open('dataset.pkl','wb'))

def make_dataset(filepath):
	list_users = os.listdir(filepath)
	load_raw_users(filepath, list_users)

if __name__=='__main__':
	#unpickled = pickle.load(open('C:\\Users\\Maria\\Desktop\\thesis\\data\\post\\emoleechen.pkl','rb'))
	#user = User(unpickled)

	#print user.get_all_hashtags()
	#print user.get_spendings()

	make_dataset('C:\\Users\\Maria\\Desktop\\thesis\\data\\post\\')