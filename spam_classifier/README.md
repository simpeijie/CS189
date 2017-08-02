# Spam Classifier 

A spam classifier modeled using logistic regression. 

spam.mat in `data/` consists of 4601 email messages, from which 57 features have been extracted as follows:
* 48 features giving the percentage (0 - 100) of words in a given message which match a given word on the list. 
The list contains words such as business, free, george, etc. (The data was collected by George Forman, so his name occurs quite a lot!)
* 6 features giving the percentage (0 - 100) of characters in the email that match a given character on the list.
The characters are ;( [ ! $ # .
* Feature 55: The average length of an uninterrupted sequence of capital letters
* Feature 56: The length of the longest uninterrupted sequence of capital letters
* Feature 57: The sum of the lengths of uninterrupted sequence of capital letters

Of all the messages, 3450 are the training set and 1151 are the test set. 

## Implementation

The dataset is prepocessed in three different ways before being used for training and testing:
1. Standardizing the columns so they have mean 0 and unit variance.
2. Transforming the features using log(x_ij + 0.1).
3. Binarizing the features on x_ij > 0.

The gradient descent equations are derived for logistic regression with L2 regularization. Gradient descent aims to minimize the negative log likelihood. The loss after each iteration is being kept track of and a graph is plotted for verification. 

Unlike gradient descent in which one iteration involves scanning through the whole training data and computing the full gradient, stochastic gradient descent processes one random data point each iteration. 

The hyperparameters which include the learning rate, the boundary for classifying if an email is spam vs. ham, and the regularization parameter are selected after some tuning. Here, the parameters are initialized to 0 and the model is trained on values ranging from 0 to 1, each with increments of 0.1. 

```python
accuracy = np.array([[[None for _ in range(range_of_reg.size)] for _ in range(range_of_alphas.size)] for _ in range(range_of_boundaries.size)])
	for b in range(range_of_boundaries.size):
		for a in range(range_of_alphas.size):
			for r in range(range_of_reg.size):
				temp_a, temp_r, temp_b = alpha + a*0.1, reg + r*0.1, boundary + b*0.1
				model = train_gd(X_train_std, y_train, temp_a, temp_r)
				preds = predict(X_train_std.dot(model), temp_b)
				accuracy[b][a][r] = metrics.accuracy_score(y_train, preds)
```