import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

""" Plot data """

def plot_counter(counter, xlabel='untitled', ylabel='untitled', title='untitled', imagename='default'):
	labels = counter.keys()
	y_pos = np.arange(len(labels))
	
	values_percentage = [val/float(sum(counter.values())) * 100 for val in counter.values()]
	values = counter.values()

	plt.barh(y_pos, values_percentage, alpha=0.4)
	plt.yticks(y_pos, labels)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)

	plt.tight_layout()
	plt.savefig(imagename+'.png')
	plt.savefig(imagename+'.pdf')
	plt.close()	

def plot(xvalues, yvalues, xlabel='untitled', ylabel='untitled', title='untitled', imagename='default'):
	y_pos = np.arange(len(yvalues))
	values = xvalues

	plt.barh(y_pos, xvalues, alpha=0.4)
	plt.yticks(y_pos, yvalues)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)

	plt.tight_layout()
	plt.savefig(imagename+'.png')
	plt.savefig(imagename+'.pdf')
	plt.close()	

def plot_matrix(matrix, xlabel='', ylabel='', title='', imagename='untitled'):

	plt.plot(matrix)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)

	plt.savefig(imagename+'.png')
	plt.close()