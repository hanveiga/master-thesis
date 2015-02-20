from plots import plot_matrix
import pickle as cPickle
import sys

error_matrix = pickle.load(open(sys.argv[1],'rb'))

count = 0
for error_slice in error_matrix:
	plot_matrix(error_slice, title='Accuracy', ylabel='Errors', xlabel='Hashtags', filename=str(count))
	count = count + 1