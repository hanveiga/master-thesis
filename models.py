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

from sklearn.linear_model import SGDRegressor

''' Wrapper aroung sklearn to vectorize Annotation objects
'''
class BasicVectorizer(object):
  def __init__(self):
    self.vectorizer = CountVectorizer()

  def fit_transform(self, users):
    return self.vectorizer.fit_transform(
      [user.get_all_hashtags() for user in users])

  def transform(self, users):
    return self.vectorizer.transform(
      [user.get_all_hashtags() for user in users])


''' Annotation classifier base class. Subclass it and specify
    a vectorizer and a classifiers
'''
class HashtagClassifier(object):
  def train(self, users):
    X = self.vectorizer.fit_transform(users)
    y = numpy.array([np.mean(filter(None,user.get_spendings())) for user in users])

    self.classifier.fit(X, y)

  def predict(self, users):
    X = self.vectorizer.transform(users)
    return self.classifier.predict(X)

class NaiveRegression(HashtagClassifier):
  def __init__(self):
    self.classifier = SGDRegressor()
    self.vectorizer = BasicVectorizer()