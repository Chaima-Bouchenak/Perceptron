# coding: utf8
# !/usr/bin/env python
# --------------------------------------------------------------------------------------
# Perceptron Algorithm implementation
# Authors : Beavogui Angelina
#           Bouchenak Chaima
# --------------------------------------------------------------------------------------
# Implementation of a simple perceptron for supervised linear classification.
#
# The perceptron is the simplest possible neural network. It lacks any hidden layers,
# and uses binary threshold as its activation function. it’s good for tasks where we
# want to predict if an input belongs in one of two categories, based on it’s features
# and the features of inputs that are known to belong to one of those two categories.
# --------------------------------------------------------------------------------------
# Matrix calculation library
import numpy as np
from random import seed
from random import randrange
from csv import reader
import matplotlib.pyplot as plt


def predict(test_inputs):
    """
     Predict the categorical class labels of inputs according to activation rate.
     @X : input vector/array
     return:
         activation rate
     """
    global y_predicted
    n_samples, n_features = test_inputs.shape

    # init parameters
    weights = np.zeros(n_features)
    bias = 0
    linear_output = np.dot(test_inputs, weights) + bias

    for idx, x_i in enumerate(test_inputs):  # iter over training samples to get x value and its index
        # predict test labels
        linear_output = np.dot(x_i, weights) + bias
        # apply activation func to predict test labels
        y_predicted = step_func(linear_output)
    return y_predicted


def fit(train_inputs, train_outputs, lr, n_iters):
    """
    Train the model by using labeled data and predict output,
    determine if it was correct or not, and then adjust the weights and bias
    accordingly. it works as the equation : w <- w + alpha * (y - f(x)) * x
    @X : list of known inputs/vectors
        = inputs in predict()
    @y : known outputs = training labels
    """
    n_samples, n_features = train_inputs.shape

    # init parameters
    weights = np.zeros(n_features)
    bias = 0

    # convert values to 0 or 1
    y_ = np.array([1 if i > 0 else 0 for i in train_outputs])

    # start training model
    for _ in range(n_iters):

        for idx, x_i in enumerate(train_inputs):  # iter over training samples to get x value and its index
            # predict test labels
            linear_output = np.dot(x_i, weights) + bias
            y_predicted = step_func(linear_output)

            # Perceptron weight update rule for each training sample
            update = lr * (y_[idx] - y_predicted)

            weights += update * x_i
            bias += update * 1


def step_func(train_outputs):
    # return 1 if x >= 0 else 0
    return np.where(train_outputs >= 0, 1, 0)  # return np array if n_inputs is >= 0

def accuracy_split(y_true, y_pred):
    """
    Determinate the fraction of instances that was predicted correctly by the
    perceptron model
    @y_true : np.array of known labels
    @y_pred : np.array of predicted labels
    """
    accuracy_rate = np.sum(y_true == y_pred) / len(y_true)
    return accuracy_rate

# Make a prediction with weights
def predict_cross_vald(line, weights):
    """
    predicts an output value for a row given a set of weights
    """
    activation = weights[0]
    for i in range(len(line) - 1):
        activation += weights[i + 1] * line[i]
    return 1.0 if activation >= 0.0 else 0.0


# Split a dataset into k folds
def k_folds_split(dataset, n_folds):
    """
    Split a dataset into k folds to estimate the model performance
    @dataset : list of the dataset
    @n_folds : number of folds to be generated
    return :
        list of n_folds
    """
    dataset_folds = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_folds.append(fold)
    return dataset_folds


def train_weights(train, l_r, n_iters):
    """Calculates coefficient values for a training dataset using stochastic gradient descent."""
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_iters):
        for line in train:
            prediction = predict_cross_vald(line, weights)
            train_error = line[-1] - prediction
            weights[0] = weights[0] + l_r * train_error
            for i in range(len(line) - 1):
                weights[i + 1] = weights[i + 1] + l_r * train_error * line[i]
    return weights


def perceptron(train, test, l_r, n_iters):
    """
    perform Perceptron Algorithm With Stochastic Gradient Descent
    """
    predictions = list()
    weights = train_weights(train, l_r, n_iters)
    for line in test:
        prediction = predict_cross_vald(line, weights)
        prediction = round(prediction)
        predictions.append(prediction)
    return predictions


def accuracy_percentage(actual, predicted):
    """
    Calculate accuracy percentage
    """
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def perceptron_performance(dataset, algorithm, n_folds, *args):
    """
    Evaluate an algorithm using a cross validation split
    """
    folds = k_folds_split(dataset, n_folds)
    acc_scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for line in fold:
            line_copy = list(line)
            test_set.append(line_copy)
            line_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_percentage(actual, predicted)
        acc_scores.append(accuracy)
    return acc_scores


def plot_results(X_train_pca, X_test_pca, y_train, y_test):
    """
    plotting the decision boundary in the scatter plot of Training and Test Set with labels indicated by colors
    """
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

    xx_train, yy_train = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                     np.arange(y_min, y_max, 0.1))

    Z_train = predict(np.c_[xx_train.ravel(), yy_train.ravel()])
    Z_train = Z_train.reshape(1,(xx_train.shape))

    x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
    y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1

    xx_test, yy_test = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                   np.arange(y_min, y_max, 0.1))

    Z_test = predict(np.c_[xx_test.ravel(), yy_test.ravel()])
    Z_test = Z_test.reshape(xx_test.shape)

    plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 1)
    plt.contourf(xx_train, yy_train, Z_train)
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, s=30, edgecolor='k')
    plt.xlabel('Training 1st Principal Component')
    plt.ylabel('Training 2nd Principal Component')
    plt.title('Scatter Plot with Decision Boundary for the Training Set')
    plt.subplot(1, 2, 2)
    plt.contourf(xx_test, yy_test, Z_test)
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, s=30, edgecolor='k')
    plt.xlabel('Test 1st Principal Component')
    plt.ylabel('Test 2nd Principal Component')
    plt.title('Scatter Plot with Decision Boundary for the Test Set')
    plt.show()


def load_datafile(filename):
    """
    Load the chosen dataset CSV file.
    @filename : name of the csv file in the current working directory
    return:
        a list of the dataset
    """
    dataset = list()
    with open(filename, 'r') as file:
        read_file = reader(file)
        for line in read_file:
            if not line:
                continue
            dataset.append(line)
    return dataset


def elem_str_to_float(dataset, column):
    """
    Convert string column to float
    """
    for line in dataset:
        line[column] = float(line[column].strip())


def elem_str_to_bool(dataset, column):
    """
    Convert string column to integer
    """
    elem_list = [line[column] for line in dataset]
    unique = set(elem_list)
    dico = dict()
    for i, value in enumerate(unique):
        dico[value] = i
    for line in dataset:
        line[column] = dico[line[column]]
    return dico
# --------------------------------------------------------------------------------------
