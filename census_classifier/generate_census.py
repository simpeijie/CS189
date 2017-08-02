import numpy as np
import csv
import pandas as pd
import scipy.io
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer

np.set_printoptions(threshold=float('inf'))

#################
# Training data #
#################

raw_training_data = []
numerical = ['age', 'fnlwgt','education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
with open('hw5_data/census_data/train_data.csv') as csvFile:
    reader = csv.DictReader(csvFile)
    
    for row in reader:
        raw_training_data.append(row)

labels = []
categories = []
numerals = []

v = DictVectorizer(sparse=False)

for datum in raw_training_data:
	tempCat, tempNum = {}, []
	for key, val in datum.items():
		if key == "label":
			labels.append(int(val))
		elif key in categorical:
			tempCat[key] = val
		elif key in numerical:
			tempNum.append(float(val))

	categories.append(tempCat)
	numerals.append(tempNum)

labels = np.array(labels).reshape((np.shape(labels)[0], 1))

catTransformed = v.fit_transform(categories)
training_data = np.array([])
training_data = np.hstack((catTransformed, numerals))
training_data = np.array([[int(i) for i in d] for d in training_data])

#############
# Test data #
#############

raw_test_data = []

with open('hw5_data/census_data/test_data.csv') as csvFile:
    reader = csv.DictReader(csvFile)
    
    for row in reader:
        raw_test_data.append(row)
        
categories = []
numerals = []

for datum in raw_test_data:
	tempCat, tempNum = {}, []
	for key, val in datum.items():
		if key in categorical:
			tempCat[key] = val
		elif key in numerical:
			tempNum.append(float(val))

	categories.append(tempCat)
	numerals.append(tempNum)

catTransformed = v.transform(categories)
test_data = np.array([])
test_data = np.hstack((catTransformed, numerals))
test_data = np.array([[int(i) for i in d] for d in test_data])

# Store in .mat file

file_dict = {}
file_dict['training_data'] = training_data
file_dict['training_labels'] = labels
file_dict['test_data'] = test_data
scipy.io.savemat('census.mat', file_dict)
