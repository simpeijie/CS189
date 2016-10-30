# Spam Classifier 

A spam classifier modeled using logistic regression. 

spam.mat in data/ consists of 4601 email messages, from which 57 features have been extracted as follows:
* 48 features giving the percentage (0 - 100) of words in a given message which match a given word on the list. 
The list contains words such as business, free, george, etc. (The data was collected by George Forman, so his name occurs quite a lot!)
* 6 features giving the percentage (0 - 100) of characters in the email that match a given character on the list.
The characters are ;( [ ! $ # .
* Feature 55: The average length of an uninterrupted sequence of capital letters
* Feature 56: The length of the longest uninterrupted sequence of capital letters
* Feature 57: The sum of the lengths of uninterrupted sequence of capital letters

Of all the messages, 3450 are the training set and 1151 are the test set. 
