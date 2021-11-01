# coding: utf8
# !/usr/bin/env python
# --------------------------------------------------------------------------------------
# Perceptron Algorithm implementation
# Authors : Beavogui Angelina
#           Bouchenak Chaima
# ------------------------------------------------------------------------------
# Implementation of a multilayer perceptron with gradient backpropagation.
# ------------------------------------------------------------------------------
# Librairie de calcul matriciel
import numpy


class MLP:
    ''' Class implementing a multilayer perceptron '''

    def __init__(self, *args):
        '''
        @summary: Creation of the network
        @param args: List of layer sizes
        @type args: tuple
        '''
        self.shape = args
        n = len(args)
        self.layers = []
        self.layers.append(numpy.ones(self.shape[0] + 1))
        # hidden and output layers
        for i in range(1, n):
            self.layers.append(numpy.ones(self.shape[i]))
        self.weights = []
        for i in range(n - 1):
            self.weights.append(numpy.zeros((self.layers[i + 1].size, self.layers[i].size)))
        # Initialisation des poids
        self.reset()

    def reset(self):
        '''
        @summary: Initialization of weights between -1 and 1
        '''
        for i in range(len(self.weights)):
            self.weights[i][:] = numpy.random.random((self.layers[i + 1].size, self.layers[i].size)) * 2. - 1.

    def propagate_forward(self, data):
        '''
        @summary: Propagation of the input through the hidden layers to the output layer
        @param data: The current input
        @type data: numpy.array
        @return: The activity of the output layer (numpy.array)
        '''
        # update inputs layer
        self.layers[0][0:-1] = data
        for i in range(1, len(self.shape)):
            self.layers[i][:] = sigmoid(numpy.dot(self.weights[i - 1], self.layers[i - 1]))
        return self.layers[-1]

    def propagate_backward(self, target, lrate):
        '''
        @summary: Update weights by gradient backpropagation
        @param target: The expected output
        @type target: numpy.array
        @param lrate: The learning rate
        @type lrate: float
        '''
        deltas = []
        # error calculation
        error = target - self.layers[-1]
        delta = error * dsigmoid(self.layers[-1])
        deltas.append(delta)
        # Rétropropagation du gradient
        for i in range(len(self.shape) - 2, 0, -1):
            delta = numpy.dot(deltas[0], self.weights[i]) * dsigmoid(self.layers[i])
            deltas.insert(0, delta)
        # weights update
        for i in range(len(self.weights)):
            inp = self.layers[i][numpy.newaxis, :]
            delta = deltas[i][:, numpy.newaxis]
            self.weights[i] += lrate * delta * inp

    def learn(self, train_samples, epochs, lrate, verbose=False):
        '''
        @summary: Model learning
        @param train_sample: Training data set
        @type train_sample: dictionary ('input' is a numpy.array which contains the set of input vectors, 'output' is a numpy.array which contains the set of corresponding output vectors)
        @param epochs: Number of learning steps
        @type epochs: int
        @param lrate: Learning rate
        @type lrate: float
        @param verbose: Indicates if the display is activated (False by default)
        @type verbose: boolean
        '''
        for i in range(epochs):
            # take a random choice
            n = numpy.random.randint(train_samples['input'].shape[0])
            self.propagate_forward(train_samples['input'][n])
            # learn by rétropropagation gradient
            error = self.propagate_backward(train_samples['output'][n], lrate)
            if verbose and i % 1000 == 0:
                print('epoch ', i, ' error ', error)

    def test_regression(self, test_samples, verbose=False):
        '''
        @summary: Test of the model (mean square error)
        @param test_samples: Test data set
        @type test_samples: dictionary ('input' is a numpy.array which contains the set of input vectors, 'output' is a numpy.array which contains the set of corresponding output vectors)
        @param verbose: Indicates if the display is enabled (False by default)
        @type verbose: boolean
        '''

        # Erreur quadratique moyenne
        error = 0.
        for i in range(test_samples['input'].shape[0]):
            # Calculation of the output corresponding to the current test example
            o = self.propagate_forward(test_samples['input'][i])
            # Update of the mean square error
            error += numpy.sum((o - test_samples['output'][i]) ** 2)
            if verbose:
                print('entree', test_samples['input'][i], 'sortie %.2f' % o,
                      '(attendue %.2f)' % test_samples['output'][i])
        print('erreur quadratique moyenne ', error / test_samples['input'].shape[0])

    def test_classification(self, test_samples):
        '''
        @summary: Model test (classification)
        @param test_samples: Test data set
        @type test_samples: dictionary ('input' is a numpy.array that contains the set of input vectors, 'output' is a numpy.array that contains the set of corresponding output vectors)
        '''
        # Number of misclassifications
        error = 0.
        for i in range(test_samples['input'].shape[0]):
            # Calculation of the output corresponding to the current test example
            o = self.propagate_forward(test_samples['input'][i])
            # Increase in the number of errors if the max of the output does not correspond to the expected category
            error += 0. if numpy.argmax(o) == numpy.argmax(test_samples['output'][i]) else 1.
        print('erreur de classification ', error / test_samples['input'].shape[0] * 100, '%')

def sigmoid(x):
    '''
    @summary: Equation of a sigmoid
    @param x: Input value
    @type x: numpy.array
    @return: The sigmoid applied to point x (numpy.array)
    '''
    return 1. / (1. + numpy.exp(-x))


def dsigmoid(x):
    '''
    @summary: Derivative of the sigmoid function
    @param x: Input value
    @type x: numpy.array
    @return: The derivative applied to point x (numpy.array)
    '''
    return x * (1. - x)
# -----------------------------------------------------------------------------
