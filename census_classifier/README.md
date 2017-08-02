# Census Classifier

Decision Trees and Random Forest for classification on a census dataset to predict if a person makes over 50k in income. 

## Data Processing

In `census_data/`, the training and test data are in csv format. They have to be processed and transformed into a form suitable for consumption by the decision tree.

The data is categorized into numerical and categorical variables. By using `DictVectorizer` from `sklearn.feature_extraction`, we can vectorize on one-hot encode those features. 

Doc here: [](http://scikit-learn.org/stable/modules/feature_extraction.html)

By running 
```
python generate_census.py
```
the data is tranformed and stored in `census.mat`.

## Node

A decision tree is a binary tree composed of Nodes. Each node contains four key parameters, namely 
* split_rule - a length 2 tuple that details what features to split on at a node and a threshold value at which the tree should split at.
* left - left child of current node.
* right - right child of current node.
* label - if this parameter is set, the node is a leaf node. 

## Decision Tree

There are four methods associated with a decision tree:
* impurity(left_label_hist, right_label_hist) - a histogram is a mapping from label values to their frequencies. This method calculates and outputs a scalar value representing the impurity (badness) of the specified split on the input data.
* segmenter(data, labels) - finds the best split rule for Node using the impurity measure and input data.
* train(data, labels) - grows the decision tree by constructing nodes.
* predict(data) - given a data point, traverse the tree to find the best label to classify the data point. 

## Training 

Create a new DecisionTree object and run `train`, passing in the training data and labels.  

## Validation

The training data is split 80:20 into training and validation sets.
