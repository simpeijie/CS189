from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import sklearn.metrics as metrics
from numpy import unravel_index
from datascience import *

np.set_printoptions(threshold=float('inf'))

NUM_CLASSES = 10
input_layer_size = 784
hidden_layer_size = 200
output_layer_size = NUM_CLASSES

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    # The test labels are meaningless,
    # since you're replacing the official MNIST test set with our own test set
    X_test, _ = map(np.array, mndata.load_testing())
    # Remember to center and normalize the data...
    return X_train, labels_train, X_test

def one_hot(labels_train):
	'''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
	return np.eye(NUM_CLASSES)[labels_train]

def shuffle(X_train, y_train):
	n, d = X_train.shape
	arr = np.random.permutation(n)
	X = np.zeros((n, d))
	y = np.zeros((n,), dtype=np.int)

	for i in range(n):
		X[i, :] = X_train[arr[i], :]
		y[i] = y_train[arr[i]]

	return X, y

def normalize(X_train, X_test):
	return (X_train - np.mean(X_train)) / np.std(X_train), \
			(X_test - np.mean(X_test))/ np.std(X_test)

def relu(z):
	return z * (z > 0)

def relu_derivative(z):
	new_z = z
	new_z[new_z < 0] = 0
	new_z[new_z > 0] = 1
	return new_z

def softmax(z):
	return np.exp(z) / np.sum(np.exp(z), axis=0)

def forwardPropagate(V, W, X):
	s_hidden = np.dot(V.T, X.reshape(X.shape[0], 1)) # (200, 785) x (785, 1) = (200, 1) 
	x_hidden = relu(s_hidden)
	x_hidden = np.append(x_hidden, [1]).reshape(s_hidden.shape[0]+1, 1) # (201, 1)

	s_output = np.dot(W.T, x_hidden) # (10, 201) x (201 x 1) = (10, 1)
	x_output = softmax(s_output) # (10, 1)
	return x_hidden, x_output, s_hidden

def cross_entropy_error(y, y_pred):
	J = -np.sum(np.multiply(y, np.log(y_pred + 1e-30)) + np.multiply((1.-y), np.log(1.-y_pred + 1e-30)))
	return J

def backwardPropagate(W, x_output, x_hidden, s_hidden, xi, y):
	delta_output = x_output - y # (10, 1)
	delta_1 = np.dot(W, delta_output) # (201, 1)
	delta_2 = relu_derivative(s_hidden)
	delta_1 = np.delete(delta_1, 200).reshape(200, 1) # (200, 1)
	delta_hidden = np.multiply(delta_1, delta_2) # (200, 1)

	return np.dot(x_hidden, delta_output.T), np.dot(xi, delta_hidden.T)
					# (201, 10)						# (785, 200)

def trainNeuralNetwork(V, W, X, y, labels, alpha=1e-3, epoch=5, gamma=0.9, num_iter=50000):
	n, d = X.shape
	rate = alpha
	error = []
	accuracy = []
	for e in range(epoch):
		rate *= gamma
		for i in range(num_iter):
			# Pick data point in order since data is already shuffled
			xi, yi = X[i, :].reshape((d, 1)), y[i].reshape((NUM_CLASSES, 1))
			x_hidden, x_output, s_hidden = forwardPropagate(V, W, xi)
			dJdW, dJdV = backwardPropagate(W, x_output, x_hidden, s_hidden, xi, yi) 
			W = W - rate * dJdW
			V = V - rate * dJdV
			if i % 10000 == 0:
				preds = predictNeuralNetwork(V, W, X)
				accuracy.append(metrics.accuracy_score(labels, preds))
				error.append(cross_entropy_error(yi, x_output))
	
	return W, V, error, accuracy

def predictNeuralNetwork(V, W, X):
	n, d = X.shape
	preds = np.zeros((n,), dtype=np.int)
	i, cost = 0, 0
	while i < n:
		xi = X[i, :].reshape((d, 1))
		_, pred, _ = forwardPropagate(V, W, xi)
		preds[i] = np.argmax(pred)
		i += 1

	return preds

if __name__ == "__main__":	
	# Pick weights at random to start
	V = 1e-3 * np.random.randn(input_layer_size + 1, hidden_layer_size) # (785, 200)
	W = 1e-3 * np.random.randn(hidden_layer_size + 1, output_layer_size) # (201, 10)
	
	# Loading, shuffling and normalizing data
	X_train, labels_train, X_test = load_dataset()
	X_train, labels_train = shuffle(X_train, labels_train)
	X_train, X_test = normalize(X_train, X_test)

	n = X_train.shape[0]
	
	# Append ones to the last column 
	ones = np.ones((n, 1))
	X_train = np.append(X_train, ones, axis=1) # (60000, 785)
	
	# Split data into training set (50000) and validation set (10000)
	X_train, X_cv = X_train[0:50000,:], X_train[50000:60000,:]
	labels_train, y_cv = labels_train[0:50000], labels_train[50000:60000]
	
	y_train = one_hot(labels_train) # (50000, 10)

	# Update n to only be 50000
	n = X_train.shape[0]

	# Parameter selection
	# for a in range(range_of_alpha.size):
	# 	temp_a = alpha + a*0.01
	# 	new_W, new_V = trainNeuralNetwork(V, W, X_train, y_train, alpha=temp_a)
	# 	train_preds = predictNeuralNetwork(new_V, new_W, X_train)
	# 	train_acc = metrics.accuracy_score(labels_train, train_preds)
	# 	cv_preds = predictNeuralNetwork(new_V, new_W, X_cv)
	# 	cv_acc = metrics.accuracy_score(y_cv, cv_preds)
	# 	print("alpha={0}:\nTrain accuracy: {1}\nCV accuracy: {2}".format(temp_a, train_acc, cv_acc))
	# max_alpha = alpha + 0.01*np.argmax(accuracy)

	###########################
	# KAGGLE SUBMISSION PHASE #
	###########################
	# Note: Without splitting data into training and validation.

	# X_train, X_test = normalize(X_train, X_test)
	# ones = np.ones((X_train.shape[0], 1))
	# X_train = np.append(X_train, ones, axis=1) # (60000, 785)
	# ones = np.ones((X_test.shape[0], 1))
	# X_test = np.append(X_test, ones, axis=1)
	# y_train = one_hot(labels_train)

	#############
	# Verifying #
	#############
	new_W, new_V, error, accuracy = trainNeuralNetwork(V, W, X_train, y_train, labels_train, alpha=0.01, epoch=5, num_iter=n)

	# Plot of error
	plt.xlabel("No. of iterations")
	plt.ylabel("Error")
	plt.plot(error)
	plt.show()

	# Plot of accuracy
	plt.xlabel("No. of iterations")
	plt.ylabel("Accuracy")
	plt.plot(accuracy)
	plt.show()

	# Predictions
	preds = predictNeuralNetwork(new_V, new_W, X_train)
	print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, preds)))
	cv_preds = predictNeuralNetwork(new_V, new_W, X_cv)
	print("CV accuracy: {0}".format(metrics.accuracy_score(y_cv, cv_preds)))

	# kag_preds = predictNeuralNetwork(new_V, new_W, X_test)
	# t = Table().with_columns([['Id', np.arange(1, X_test.shape[0]+1)], ['Category', kag_preds]])
	# t.to_csv('hw4-kaggle-testing-nov03.csv')




