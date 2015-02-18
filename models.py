''' Classifiers and Vectorizers here are wrappers around
    sklearn objects that classify and vectorize annotations,
    respectively
'''

import random
import numpy
import operator
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import itertools

from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB

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

class RawVectorizer(object):
  def __init__(self):
    self.vectorizer = CountVectorizer()

  def fit_transform(self, users):
    return self.vectorizer.fit_transform(
      [ ' '.join(user.get_all_hashtags()) for user in users])

  def transform(self, users):
    return self.vectorizer.transform(
      [' '.join(user) for user in users])


''' Annotation classifier base class. Subclass it and specify
    a vectorizer and a classifiers
'''
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
  def train(self, users, labels):
    X = self.vectorizer.fit_transform(users)

    self.classifier.fit(X.toarray(), labels)

  def predict(self, users):
    X = self.vectorizer.transform(users)
    return self.classifier.predict(X.toarray())

class NaiveClassifier(VenueClassifier):
  def __init__(self):
    self.classifier = GaussianNB()
    self.vectorizer = BasicVectorizer()

class ProgressiveClassifier(VenueClassifier):
  def __init__(self):
    self.classifier = GaussianNB()
    self.vectorizer = RawVectorizer()

  def predict(self,users):
    X = self.vectorizer.transform(users)
    return self.classifier.predict(X.toarray())

''' Generating labels '''
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