from sklearn.metrics.pairwise import cosine_distances
from scipy.sparse import find

def cosine_similarity(vector_a, vector_b):
	return 1-cosine_distances(vector_a,vector_b)

def similarity(vector_a, vector_b):
	# check if vector b is in a
	_, features_a , _ = find(vector_a)
	_, features_b , _ = find(vector_b)

	set_a = set(features_a)
	set_b = set(features_b)

	#print set_a
	#print set_b

	new_information = set_b - set_a 

	#/ len(vector_b)
	if len(set_b) is not 0:
		tweet_novelty = len(new_information)/float(len(set_b))
	else:
		tweet_novelty = 0

	return tweet_novelty