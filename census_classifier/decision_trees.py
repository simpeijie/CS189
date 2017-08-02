import numpy as np 
import scipy.io
import random
import sklearn.metrics as metrics
from numpy import unravel_index
from math import sqrt
from scipy.stats import itemfreq
from datascience import *

np.set_printoptions(threshold=float('inf'))

class Node:
	def __init__(self, height, labels_freq, split_rule=None, left=None, right=None, is_label=False):
		self.height = height
		self.labels_freq = labels_freq
		self.split_rule = split_rule
		self.left = left
		self.right = right
		self.is_label = is_label

	def get_label(self):
		return max(self.labels_freq, key=lambda x: x[1])[0]

class DecisionTree:
	def __init__(self, max_height, trees=1, rf=False):
		self.max_height = max_height
		self.rf = rf
		self.trees = trees
		self.nodes = []
		self.root = None

	def entropy(self, prob):
		return -np.sum(prob*np.log(prob))

	def impurity(self, left_label_hist, right_label_hist):
		left_count, right_count = left_label_hist[:, 1], right_label_hist[:, 1]
		left_total, right_total = np.sum(left_count), np.sum(right_count)
		left_entropy, right_entropy = self.entropy(left_count/left_total), self.entropy(right_count/right_total)

		return (left_total*left_entropy + right_total*right_entropy) / (left_total + right_total)

	def segmenter(self, data, labels):
		best_impurity = float('inf')
		best_left, best_right, best_rule = None, None, None
		data_size, num_features = data.shape

		# Random forest
		if self.rf:
			for i in range(self.trees):
				m = random.sample(range(num_features), 10)
				for j in range(num_features):
					feature = data[:, j]
					n = random.sample(range(data_size), int(data_size/50))
					for val in n:
						left_indices = np.nonzero(feature < val)[0] 
						right_indices = np.nonzero(feature >= val)[0]
						split_rule = (j, val)
						left_labels = labels[left_indices]
						right_labels =  labels[right_indices]
						if left_labels.size == 0 or right_labels.size == 0:
							continue
						impurity = self.impurity(itemfreq(left_labels), itemfreq(right_labels))
						if impurity < best_impurity:
							best_impurity = impurity
							best_rule = split_rule 
							best_left = left_indices
							best_right = right_indices
		# Normal DT
		else:
			for i in range(num_features):
				feature = data[:, i]
				for val in feature:
				# mean = np.mean(feature)
				# left_indices = np.nonzero(feature < mean)[0]
				# right_indices = np.nonzero(feature >= mean)[0]
				# split_rule = (i, mean)
					left_indices = np.nonzero(feature < val)[0] 
					right_indices = np.nonzero(feature >= val)[0]
					split_rule = (i, val)
					left_labels = labels[left_indices]
					right_labels =  labels[right_indices]
					impurity = self.impurity(itemfreq(left_labels), itemfreq(right_labels))
					if left_labels.size == 0 or right_labels.size == 0:
						continue
					if impurity < best_impurity:
						best_impurity = impurity
						best_rule = split_rule 
						best_left = left_indices 
						best_right = right_indices

		return best_rule, best_left, best_right

	def grow_tree(self, height, data, labels):
		if height == self.max_height:
			node = Node(height=height, labels_freq=itemfreq(labels), is_label=True)
			self.nodes.append(node)
			return node

		split_rule, left_indices, right_indices = self.segmenter(data, labels)
		if not split_rule:
			node = Node(height=height, labels_freq=itemfreq(labels), is_label=True)
			self.nodes.append(node)
			return node

		left_data, right_data = data[left_indices], data[right_indices]
		left_labels, right_labels = labels[left_indices], labels[right_indices]

		node = Node(height=height, \
					labels_freq=itemfreq(labels), \
					split_rule=split_rule, \
					left=self.grow_tree(height+1, left_data, left_labels), \
					right=self.grow_tree(height+1, right_data, right_labels),\
					is_label=False)

		self.nodes.append(node)
		return node

	def train(self, data, labels):
		self.root = self.grow_tree(0, data, labels) 

	def predict(self, data):
		preds = np.zeros(data.shape[0], dtype=np.int)
		for i in range(data.shape[0]):
			node = self.root
			while not node.is_label:
				datum = data[i]
				index, val = node.split_rule
				if datum[index] < val:
					node = node.left
				else:
					node = node.right
			preds[i] = node.get_label()
		return preds

def get_training_validation(data, labels):
	size = np.shape(data)
	indices = np.random.permutation(size[0])
	new_data = np.zeros(size)
	new_labels = np.zeros(size[0])
	for i in range(size[0]):
		new_data[i, :] = data[indices[i], :]
		new_labels[i] = labels[indices[i]]

	# 80:20
	train_size = int(size[0] * 0.8)

	return new_data[:train_size, :], new_labels[:train_size], \
			new_data[train_size:, :], new_labels[train_size:]

##########
#  Spam  #
##########

# mat_dict = scipy.io.loadmat("hw5_data/spam_data/spam_data.mat", appendmat=False)
# training_data = mat_dict['training_data']
# training_labels = mat_dict['training_labels'][0]
# training_data, training_labels, validation_data, validation_labels = get_training_validation(training_data, training_labels)

###########
#  Census #
###########

mat_dict = scipy.io.loadmat("census.mat", appendmat=False)
training_data = mat_dict['training_data']
training_labels = mat_dict['training_labels']
training_data, training_labels, validation_data, validation_labels = get_training_validation(training_data, training_labels)

#################
#     Train     #
#      and      #
#   Validation  #
#################

classifier = DecisionTree(5, 1, True)
classifier.train(training_data, training_labels)
predictions = classifier.predict(training_data)
print("Train accuracy: {0}".format(metrics.accuracy_score(training_labels, predictions)))

val_predictions = classifier.predict(validation_data)
print("Validation accuracy: {0}".format(metrics.accuracy_score(validation_labels, val_predictions)))

#######################
# Parameter selection #
#######################

# for h in [5, 10, 15, 20, 25]:
# 	# for t in [10, 30]:
# 		print("Height: {0}, Num_trees: {1}".format(h, 10))
# 		classifier = DecisionTree(h, 10, True)
# 		classifier.train(training_data, training_labels)
# 		predictions = classifier.predict(training_data)
# 		print("Train accuracy: {0}".format(metrics.accuracy_score(training_labels, predictions)))
# 		val_predictions = classifier.predict(validation_data)
# 		print("Validation accuracy: {0}".format(metrics.accuracy_score(validation_labels, val_predictions)))

#######################
#  Kaggle submission  #
#######################

# test_data = mat_dict['test_data']
# kag_preds = classifier.predict(test_data)
# t = Table().with_columns([['Id', np.arange(1, test_data.shape[0]+1)], ['Category', kag_preds]])
# t.to_csv('census-rf-15-fulltrain.csv')

