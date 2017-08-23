# function to load CIFAR_10 to be processed by the classifier!
# Use this in the function that calls the classifier during hyper-parameter
# tuning with spearmint!

import numpy as np
import random
from f17cs7643.get_cifar10 import load_CIFAR10

def load_cifar10_train_val():
	cifar10_dir = 'f17cs7643/data/cifar-10-batches-py'
	X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

	# load the data - just like we did before with SVM
	# Subsample the data for more efficient code execution in this exercise.
	num_training = 49000
	num_validation = 1000
	num_test = 1000

	# Our validation set will be num_validation points from the original
	# training set.
	mask = range(num_training, num_training + num_validation)
	X_val = X_train[mask]
	y_val = y_train[mask]

	# Our training set will be the first num_train points from the original
	# training set.
	mask = range(num_training)
	X_train = X_train[mask]
	y_train = y_train[mask]

	# We use the first num_test points of the original test set as our
	# test set.
	mask = range(num_test)
	X_test = X_test[mask]
	y_test = y_test[mask]

	print 'Train, validation and testing sets have been created as \n X_i and y_i where i=train,val,test'

	# Preprocessing: reshape the image data into rows
	X_train = np.reshape(X_train, (X_train.shape[0], -1))
	X_val = np.reshape(X_val, (X_val.shape[0], -1))
	X_test = np.reshape(X_test, (X_test.shape[0], -1))

	mean_image = np.mean(X_train, axis=0)

	# second: subtract the mean image from train and test data
	X_train -= mean_image
	X_val -= mean_image
	X_test -= mean_image

 	X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))]).T
	X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T
	# you need not return test data!
	return X_train, y_train, X_val, y_val, X_test, y_test

