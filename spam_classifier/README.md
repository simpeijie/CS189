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

The gradient descent equations are derived for logistic regression with L2 regularization. Gradient descent aims to minimize the negative log likelihood, and the hyperparameters which include the learning rate, the boundary for classifying if an email is spam vs. ham, and the regularization parameter are selected after some tuning. Here, the parameters are initialized to 0 and the model is trained on values ranging from 0 to 1, each with increments of 0.1. 

![parameter_tuning](img/parameter_tuning)

Unlike gradient descent in which one iteration involves scanning through the whole training data and computing the full gradient, 
stochastic gradient descent processes one data point each iteration. 
