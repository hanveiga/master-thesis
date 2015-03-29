''' Classifiers and Vectorizers here are wrappers around
    sklearn objects that classify and vectorize annotations,
    respectively
'''

import random
import numpy
import operator
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import OneClassSVM
import itertools
import re
import string
from nltk.stem import WordNetLemmatizer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier

''' Wrapper aroung sklearn to vectorize Annotation objects
'''
class BasicVectorizer(object):
  def __init__(self):
    self.vectorizer = CountVectorizer()

  def fit_transform(self, users):
    return self.vectorizer.fit_transform(
      [ ' '.join(user.get_all_hashtags()) for user in users])

  def transform(self, users):
    return self.vectorizer.transform(
      [' '.join(user.get_all_hashtags()) for user in users])

class BasicTweetVectorizer(object):
  def __init__(self):
    self.vectorizer = TfidfVectorizer()

  def fit_transform(self, users):
    join_tweets = []

    for user in users:
      join_tweets.append([''.join(remove_tweet_noise(tweet.text)) for tweet in user.twitter])

    return self.vectorizer.fit_transform([''.join(usertweets) for usertweets in join_tweets])

  def transform(self, users):
    join_tweets = []

    for user in users:
      join_tweets.append([''.join(remove_tweet_noise(tweet.text)) for tweet in user.twitter])

    return self.vectorizer.transform([''.join(usertweets) for usertweets in join_tweets])

class RawVectorizer(object):
  def __init__(self):
    self.vectorizer = TfidfVectorizer()

  def fit_transform(self, users):
    return self.vectorizer.fit_transform(
      [ ' '.join(user.get_all_hashtags()) for user in users])

  def transform(self, users):
    return self.vectorizer.transform(
      [' '.join(user) for user in users])


class TweetsVectorizer(object):
  def __init__(self):
    self.vectorizer = TfidfVectorizer()

  def fit_transform(self, users):
    join_tweets = []

    for user in users:
      join_tweets.append([''.join(remove_tweet_noise(tweet.text)) for tweet in user.twitter])

    return self.vectorizer.fit_transform([''.join(usertweets) for usertweets in join_tweets])

  def transform(self, users_list_of_tweets):
    join_tweets = []

    for user in users_list_of_tweets:
      join_tweets.append([''.join(remove_tweet_noise(tweet.text)) for tweet in user])

    return self.vectorizer.transform( [''.join(usertweets) for usertweets in join_tweets] )

class TweetsTruncatedVectorizer(object):
  def __init__(self):
    self.vectorizer = TfidfVectorizer()

  def fit_transform(self, users):
    join_tweets = []

    for user in users:
      join_tweets.append([''.join(remove_tweet_noise(tweet.text)) for tweet in user.twitter])

    return self.vectorizer.fit_transform([''.join(usertweets) for usertweets in join_tweets])

  def transform(self, users_list_of_tweets):
    join_tweets = []

    for user in users_list_of_tweets:
      join_tweets.append([''.join(remove_tweet_noise(tweet.text)) for tweet in user])

    return self.vectorizer.transform( [''.join(usertweets) for usertweets in join_tweets] )

''' Annotation classifier base class. Subclass it and specify
    a vectorizer and a classifiers
'''

class TweetsLemmatizedVectorizer(TweetsTruncatedVectorizer):
  def __init__(self):
    self.vectorizer = TfidfVectorizer(stop_words='english',min_df=5)
    self.wordnet = WordNetLemmatizer()

  def fit_transform(self, users):
    join_tweets = []
    
    for user in users:
      timeline = [''.join(remove_tweet_noise(tweet.text)) for tweet in user.twitter]
      #timeline_insta = [''.join(remove_tweet_noise(insta.text)) for insta in user.instagram]
      #print timeline_insta
      #timeline = timeline + timeline_insta
      lemmatized = []
      for tweet in timeline:
        lemma = [self.wordnet.lemmatize(word) for word in tweet.split()]
        lemmatized.append(' '.join(lemma))
      
      join_tweets.append(''.join(lemmatized))

    return self.vectorizer.fit_transform([usertweets for usertweets in join_tweets])

  def transform(self, users):
    join_tweets = []
    
    for user in users:
      timeline = [''.join(remove_tweet_noise(tweet.text)) for tweet in user.twitter]
      lemmatized = []
      for tweet in timeline:
        lemma = [self.wordnet.lemmatize(word) for word in tweet.split()]
        lemmatized.append(' '.join(lemma))
      
      join_tweets.append(''.join(lemmatized))

    return self.vectorizer.transform([usertweets for usertweets in join_tweets])

class TweetsInstagramLemmatizedVectorizer(TweetsTruncatedVectorizer):
  def __init__(self):
    self.vectorizer = TfidfVectorizer(stop_words='english',min_df=5)
    self.wordnet = WordNetLemmatizer()

  def fit_transform(self, users):
    join_tweets = []
    
    for user in users:
      timeline = [''.join(remove_tweet_noise(tweet.text)) for tweet in user.twitter]
      timeline_insta = [''.join(remove_tweet_noise(insta.text)) for insta in user.instagram]
      #print timeline_insta
      timeline = timeline + timeline_insta
      lemmatized = []
      for tweet in timeline:
        lemma = [self.wordnet.lemmatize(word) for word in tweet.split()]
        lemmatized.append(' '.join(lemma))
      
      join_tweets.append(''.join(lemmatized))

    return self.vectorizer.fit_transform([usertweets for usertweets in join_tweets])

  def transform(self, users):
    join_tweets = []
    
    for user in users:
      timeline = [''.join(remove_tweet_noise(tweet.text)) for tweet in user.twitter]
      timeline_insta = [''.join(remove_tweet_noise(insta.text)) for insta in user.instagram]
      #print timeline_insta
      timeline = timeline + timeline_insta
      lemmatized = []
      for tweet in timeline:
        lemma = [self.wordnet.lemmatize(word) for word in tweet.split()]
        lemmatized.append(' '.join(lemma))
      
      join_tweets.append(''.join(lemmatized))

    return self.vectorizer.transform([usertweets for usertweets in join_tweets])

class TweetsLemmatizedProgressiveVectorizer(TweetsLemmatizedVectorizer):
  def __init__(self):
    self.vectorizer = TfidfVectorizer(stop_words='english',min_df=5)
    self.wordnet = WordNetLemmatizer()

  def transform(self, users_list_of_tweets):
    join_tweets = []
    
    for user in users_list_of_tweets:
      timeline = [''.join(remove_tweet_noise(tweet.text)) for tweet in user]
      lemmatized = []
      for tweet in timeline:
        lemma = [self.wordnet.lemmatize(word) for word in tweet.split()]
        lemmatized.append(' '.join(lemma))
      
      join_tweets.append(''.join(lemmatized))

    return self.vectorizer.transform([usertweets for usertweets in join_tweets])



class SpendingsClassifier(object):
  def train(self, users):
    X = self.vectorizer.fit_transform(users)
    y = numpy.array([numpy.mean(filter(None,user.get_spendings())) for user in users])

    self.classifier.fit(X, y)

  def predict(self, users):
    X = self.vectorizer.transform(users)
    print X
    return self.classifier.predict(X)

class NaiveRegression(SpendingsClassifier):
  def __init__(self):
    self.classifier = LinearRegression()
    self.vectorizer = BasicVectorizer()

''' One class classifier base class.
    Given features, give probability of observing the visit or not.
    Binary classifier
'''

class VenueClassifier(object):

  #@profile
  def train(self, users, labels):
    users = self.vectorizer.fit_transform(users)

    self.classifier.fit(users.toarray(), labels)

  def predict(self, users):
    X = self.vectorizer.transform(users)
    return self.classifier.predict(X.toarray())

class VenueLogisticRegression(VenueClassifier):
  def __init__(self):
    self.classifier = LogisticRegression(class_weight='auto', penalty='l1')
    self.vectorizer = TweetsLemmatizedVectorizer()

class NaiveClassifier(VenueClassifier):
  def __init__(self):
    self.classifier = AdaBoostClassifier(n_estimators = 100, learning_rate=1.0)
    self.vectorizer = BasicVectorizer()

class NaiveClassifierTweet(VenueClassifier):
  def __init__(self):
    self.classifier = GaussianNB()
    self.vectorizer = BasicTweetVectorizer()

class SVMClassifierTweet(VenueClassifier):
  def __init__(self):
    self.classifier = OneClassSVM(nu=0.1,kernel='rbf')
    self.vectorizer = BasicTweetVectorizer()

class EnsembleClassifierTweet(VenueClassifier):
  def __init__(self):
    self.classifier = AdaBoostClassifier(n_estimators = 100, learning_rate=1.0)
    self.vectorizer = BasicTweetVectorizer()    

class EnsembleClassifierFreeVectorizerTweet(VenueClassifier):
  def __init__(self,vectorizer):
    self.classifier = AdaBoostClassifier(n_estimators = 100, learning_rate=1.0)
    self.vectorizer = vectorizer

class RFClassifierFreeVectorizerTweet(VenueClassifier):
  def __init__(self,vectorizer):
    self.classifier = ExtraTreesClassifier(n_estimators = 100, n_jobs=-1)
    self.vectorizer = vectorizer

class ProgressiveClassifier(VenueClassifier):
  def __init__(self):
    self.classifier = GaussianNB()
    self.vectorizer = RawVectorizer()

  def predict(self,users):
    X = self.vectorizer.transform(users)
    return self.classifier.predict(X.toarray())


class ProgressiveTweetClassifier(VenueClassifier):
  def __init__(self):
    self.classifier = GaussianNB()
    self.vectorizer = TweetsTruncatedVectorizer()

class ProgressiveEnsembleTweetClassifier(VenueClassifier):
  def __init__(self):
    self.classifier = AdaBoostClassifier(n_estimators = 100, learning_rate=1.0)
    self.vectorizer = TweetsLemmatizedProgressiveVectorizer()

  #@profile
  def predict(self,list_of_tweets):
    list_of_tweets = self.vectorizer.transform(list_of_tweets)
    return self.classifier.predict(list_of_tweets.toarray())


''' 
    Generating labels
'''
def get_visited_venue_labels(dataset, venue_type):
  """ If user has visited venue_type, label = 1, otherwise = 0 """

  venue_labels = []

  for user in dataset:
    user_visited = 0
    for checkin in user.foursquare:
        if venue_type in checkin.lowest_type:
          user_visited = 1
          break
        else:
          pass
    if user_visited == 1:
      venue_labels.append(1)
    else:
      venue_labels.append(0)

  return venue_labels

def remove_tweet_noise(tweet_text):
  # remove mentions:
  mentions = re.findall(r'@\S+', tweet_text)
  links = re.findall(r'https?://[^\s]+|www\.[^\s]+', tweet_text)
  #links = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.[^\s<>"]+', tweet_text)
  tweet_words = tweet_text.split()
  
  tweet_text_denoised = ' '.join([word for word in tweet_words if word not in links+mentions])

  tweet_text_denoised_punctuation = ' '.join([word for word in re.split('\W+',tweet_text_denoised) if word not in string.punctuation])
  #gross
  #tweet_text_denoised_punctuation = re.sub('[%s]' % re.escape(string.punctuation), '', tweet_text_denoised)

  return tweet_text_denoised_punctuation