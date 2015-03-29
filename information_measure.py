from sklearn.metrics.pairwise import cosine_distances

def cosine_similarity(vector_a, vector_b):
	return 1-cosine_distances(vector_a,vector_b)
