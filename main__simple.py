from numpy import arange, meshgrid
from numpy.ma import hstack

from perceptron__simple import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA


print("# --------------------------------------------------------------------------------------")
print("# Prepare the pima-indians-diabetes dataset for modeling")
print("# --------------------------------------------------------------------------------------")
# The dataset file is in the directory ../project perceptron

seed(1)  # generate the random number generator, to get same random values
# reading the  pima-indians-diabetes Data Set
filename = "pima-indians-diabetes.csv"
dataset = load_datafile(filename)
for i in range(len(dataset[0]) - 1):
    elem_str_to_float(dataset, i)

# Label Encoding of the Target Variable to 0 or 1
elem_str_to_bool(dataset, len(dataset[0]) - 1)

print("the dataset was loaded successfully, now it contains only numeric elements and ready to be trained\n")

print("# --------------------------------------------------------------------------------------")
print("#  Data exploration")
print("# --------------------------------------------------------------------------------------")

# # Get some information about the data:
dataset_df = pd.DataFrame(dataset)
print("The pima indian diabetes dataset has {0[0]} rows and {0[1]} columns\n".format(dataset_df.shape))
# print(dataset_df)

# Creating the Feature and Target Label sets
X = dataset_df.iloc[:, 1:]  # split into X (features)
y = dataset_df.iloc[:,-1]  # split into y (target label)

# Split data into training and test subsets, 70-30
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size = 0.7,
                                                    test_size = 0.3, random_state = 1234)

# feature scaling of the features in Training and Test Set
columns = X_train.columns
scalerx = StandardScaler()
X_train_scaled = scalerx.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns = columns)

X_test_scaled = scalerx.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = columns)

# Principal Component Analysis (PCA) to reduce the dimensionality of the data into 2D in both Training and Test Set
# Incremental Principal Component Analysis to select 2 features such that they explain as much variance as possible
pca = IncrementalPCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("# --------------------------------------------------------------------------------------")
print("#  Modeling Data")
print("# --------------------------------------------------------------------------------------")
seed(1)
l_rate = 10
n_epoch = 1

print("\n### evaluate algorithm by simple train/test split")
# train the model
fit(X_train_pca, y_train, l_rate, n_epoch)

# make prediction
predictions = predict(X_test_pca)
print("Perceptron classification accuracy rate in spliced data is : {:0.2f}%".format(int(accuracy_split(y_test, predictions)*100) , "%"))

# plot_results(X_train_pca, X_test_pca, y_train, y_test)
# plotting the decision boundary in the scatter plot of Training and Test Set with labels indicated by colors

print("\n### evaluate algorithm by Cross validation methode")
n_folds = 5

scores = perceptron_performance(dataset, perceptron, n_folds, l_rate, n_epoch)
print('** Accuracy rates of each set of folders are : %s' % scores)
print('** The perceptron algorithm mean Accuracy after K-folds cross validation is : %.3f%%' % (sum(scores) / float(len(scores))))

