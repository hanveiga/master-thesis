''' Classifiers and Vectorizers here are wrappers around
    sklearn objects that classify and vectorize annotations,
    respectively
'''
import numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import cPickle as pickle
from models import remove_tweet_noise, ProgressiveEnsembleTweetClassifier
from information_measure import cosine_similarity, similarity
''' Wrapper aroung sklearn to vectorize Annotation objects
'''

# Vectorizer
class LemmatizedStandAloneVectorizer(object):
  def __init__(self, trained_vectorizer):
    self.wordnet = WordNetLemmatizer()
    self.vectorizer = trained_vectorizer

  def transform(self, list_of_captions):

    lemmatized = []
    for caption in list_of_captions:
      #print caption.text
      lemma = [self.wordnet.lemmatize(word) for word in caption.text.split()]
      #print 'lemmatized: %s' %(lemma)
      lemmatized.append(' '.join(lemma))

    #print 'all together: %s' %lemmatized

    long_string = ' '.join(lemmatized)
    #print 'long string %s' %long_string

    return self.vectorizer.vectorizer.transform([long_string])

class ClassifierStandAlone(object):
  def __init__(self, trained_classifier):
    self.classifier = trained_classifier

  def predict(self, vector):
    return self.classifier.classifier.predict(vector)

def novelty(vectora, vectorb):
  """ novelty of b wrt to a"""
  diff = vectorb-vectora
  print diff 

if __name__ == '__main__':
  dataset = pickle.load(open('dataset_prunned.pkl','rb'))

  # Train classifier:
  classifier = ProgressiveEnsembleTweetClassifier()
  ylabels = [0,1,0,0,0,0,1,0,1,0,0,1,0,0,0,0,1,0,1,0]
  classifier.train([user for user in dataset[0:20]],ylabels)

  # Test classifier
  user_to_test = dataset[100]

  print user_to_test.twitter[0].text
  vectorizer = LemmatizedStandAloneVectorizer(classifier.vectorizer)
  print 'vectorized %s' %vectorizer.transform([user_to_test.twitter[0]])

  for tweet in user_to_test.twitter[0:10]:
    print tweet.text
  
  test_vector = vectorizer.transform([user_to_test.twitter[0]])
  print 'test vector 1'
  print test_vector

  test_vector2 = vectorizer.transform(user_to_test.twitter[1:5])
  print 'test vector 2'
  print test_vector2

  test_vector3 = vectorizer.transform(user_to_test.twitter[0:5])

  print 1-cosine_similarity(test_vector,test_vector2)
  print 1-cosine_similarity(test_vector3,test_vector2)
  print 1-cosine_similarity(test_vector,test_vector3)
  print similarity(test_vector, test_vector2)
  print similarity(test_vector,test_vector)
  print similarity(test_vector2,test_vector3)
  print similarity(test_vector3,test_vector)
  novelty(test_vector,test_vector)
  novelty(test_vector2,test_vector)
  novelty(test_vector2,test_vector3)
  #classifier_standalone = ClassifierStandAlone(classifier)
  #print classifier_standalone.predict(test_vector.toarray())