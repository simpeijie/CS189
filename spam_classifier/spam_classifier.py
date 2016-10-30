import numpy as np 
import scipy.io
import matplotlib.pyplot as plt
import random
import sklearn.metrics as metrics
from random import randint
from numpy import unravel_index
import pandas as pd

np.set_printoptions(threshold=float('inf'))

def standardize(X_train, X_test):
	return ((X_train - np.mean(X_train, axis=0))/ np.std(X_train, axis=0), 
			(X_test - np.mean(X_test, axis=0))/ np.std(X_test, axis=0))

def transform(X_train, X_test):
	return (np.log(X_train + 0.1), np.log(X_test + 0.1))

def binarize(X_train, X_test):
	return (np.array([[1 if X_train[i][j] > 0 else 0 for j in range(X_train.shape[1])] for i in range(X_train.shape[0])]),
			np.array([[1 if X_train[i][j] > 0 else 0 for j in range(X_test.shape[1])] for i in range(X_test.shape[0])]))

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def compute(x, y, n, d):
	y = y.reshape(n, 1)

	beta = np.ones(d).reshape(d, 1)
	mu = sigmoid(x.dot(beta))

	return x, y, beta, mu

def train_gd(X_train, y_train, alpha, reg, num_iter=10000):
	n, d = X_train.shape
	x, y, beta, mu = compute(X_train, y_train, n, d)
	loss = []
	for i in range(num_iter):
		dl = 2*reg*(beta) - x.T.dot(y-mu)
		beta = beta - (alpha/n)*dl
		mu = sigmoid(x.dot(beta))
		loss.append((1/n)*(reg*(beta.T.dot(beta)) - (y.T.dot(np.log(mu+1e-30)) + (1-y).T.dot(np.log(1-mu+1e-30))))[0][0])
	return beta, loss

def train_sgd(X_train, y_train, alpha, reg, num_iter=10000, decrease=False):
	n, d = X_train.shape
	x, y, beta, mu = compute(X_train, y_train, n, d)
	loss = []
	for i in range(num_iter):
		s = randint(0, n-1)
		xi, yi, mui = x[s, :].reshape((d,1)), y[s], mu[s]
		dl = 2*reg*beta - xi*(yi-mui)
		if decrease:
			beta = beta - 2*(alpha/((i+1)*n))*dl
		else:
			beta = beta - (alpha/n)*dl
		mu = sigmoid(x.dot(beta))
		loss.append((1/n)*(reg*(beta.T.dot(beta)) - (y.T.dot(np.log(mu+1e-30)) + (1-y).T.dot(np.log(1-mu+1e-30))))[0][0])
	return beta, loss

def predict(preds, boundary):
	return [1 if p > boundary else 0 for p in preds]


if __name__ == "__main__":
	mat_dict = scipy.io.loadmat("spam.mat", appendmat=False)
	X_train = mat_dict['Xtrain']
	X_test = mat_dict['Xtest']
	y_train = mat_dict['ytrain']

	X_train_std, X_test_std = standardize(X_train, X_test)
	X_train_trans, X_test_trans = transform(X_train, X_test)
	X_train_bin, X_test_bin = binarize(X_train, X_test)

	####################
	# gradient descent #
	####################

	# standardized x
	alpha = 0.1
	reg = 0.1
	model, loss = train_gd(X_train_std, y_train, alpha, reg)

	plt.figure(1)
	plt.xlabel("No. of iterations")
	plt.ylabel("Loss")
	plt.plot(loss)

	# transformed x
	alpha = 0.01
	reg = 0.1
	model, loss = train_gd(X_train_trans, y_train, alpha, reg)

	plt.figure(2)
	plt.xlabel("No. of iterations")
	plt.ylabel("Loss")
	plt.plot(loss)

	# binarized x
	alpha = 0.1
	reg = 0.1
	model, loss = train_gd(X_train_bin, y_train, alpha, reg)

	plt.figure(3)
	plt.xlabel("No. of iterations")
	plt.ylabel("Loss")
	plt.plot(loss)

	###############################
	# stochastic gradient descent #
	###############################

	standardized x
	alpha = 1
	reg = 1
	model, loss = train_sgd(X_train_std, y_train, alpha, reg)

	plt.figure(4)
	plt.xlabel("No. of iterations")
	plt.ylabel("Loss")
	plt.plot(loss)

	# transformed x
	alpha = 1
	reg = 1
	model, loss = train_sgd(X_train_trans, y_train, alpha, reg)

	plt.figure(5)
	plt.xlabel("No. of iterations")
	plt.ylabel("Loss")
	plt.plot(loss)

	# binarized x
	alpha = 1
	reg = 1
	model, loss = train_sgd(X_train_bin, y_train, alpha, reg)

	plt.figure(6)
	plt.xlabel("No. of iterations")
	plt.ylabel("Loss")
	plt.plot(loss)

	####################################
	# stochastic with decreasing alpha #
	####################################
	# standardized x
	alpha = 150
	reg = 1
	model, loss = train_sgd(X_train_std, y_train, alpha, reg, decrease=True)

	plt.figure(7)
	plt.xlabel("No. of iterations")
	plt.ylabel("Loss")
	plt.plot(loss)

	# transformed x
	alpha = 100
	reg = 1
	model, loss = train_sgd(X_train_trans, y_train, alpha, reg, decrease=True)

	plt.figure(8)
	plt.xlabel("No. of iterations")
	plt.ylabel("Loss")
	plt.plot(loss)

	# binarized x
	alpha = 100
	reg = 1
	model, loss = train_sgd(X_train_bin, y_train, alpha, reg, decrease=True)

	plt.figure(9)
	plt.xlabel("No. of iterations")
	plt.ylabel("Loss")
	plt.plot(loss)


	plt.show()

	###############
	# predictions #
	###############
	boundary = 0
	alpha = 0
	reg = 0

	range_of_boundaries = np.arange(boundary, 1, 0.1)
	range_of_alphas = np.arange(alpha, 1, 0.1)
	range_of_reg = np.arange(reg, 1, 0.1)

	accuracy = np.array([[[None for _ in range(range_of_reg.size)] for _ in range(range_of_alphas.size)] for _ in range(range_of_boundaries.size)])
	for b in range(range_of_boundaries.size):
		for a in range(range_of_alphas.size):
			for r in range(range_of_reg.size):
				temp_a, temp_r, temp_b = alpha + a*0.1, reg + r*0.1, boundary + b*0.1
				model = train_gd(X_train_std, y_train, temp_a, temp_r)
				preds = predict(X_train_std.dot(model), temp_b)
				accuracy[b][a][r] = metrics.accuracy_score(y_train, preds)

	tup = unravel_index(np.argmax(accuracy), accuracy.shape)
	max_boundary = boundary + tup[0]*0.1
	max_alpha = alpha + tup[1]*0.1
	max_reg = reg + tup[2]*0.1

	print("max alpha: ", max_alpha)
	print("max reg: ", max_reg)
	print("max boundary: ", max_boundary)

	# verifying
	model = train_gd(X_train_std, y_train, max_alpha, max_reg)
	model = train_gd(X_train_std, y_train, 0.4, 0.1)
	preds = predict(X_train_std.dot(model), 0.9)

	preds = predict(X_train_std.dot(model), max_boundary)
	print("Train accuracy: {0}".format(metrics.accuracy_score(y_train, preds)))

	preds = predict(X_test_std.dot(model), boundary)

	# pd.DataFrame(preds).to_csv('kaggle-10-11.csv')




