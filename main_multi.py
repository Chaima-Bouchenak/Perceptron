from perceptron_multi import MLP
import numpy
# Creation of a network with two inputs, 5 neurons in the hidden layer and 1 output neuron
network = MLP(2, 5, 1)

# Exemple 1 : OR function
# -------------------------------------------------------------------------
print("===============================")
print("learning the OR function")
print("===============================")
# Creation of training data
train_input = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_output = numpy.array([[0], [1], [1], [1]])
train = {'input': train_input, 'output': train_output}

# tests data are the same
test = train.copy()

# Network initialization
network.reset()
print("Performance before learning")
network.test_regression(test, True)
# learn the network
network.learn(train, 50000, 0.1, False)

print("\nperformance after learning")
network.test_regression(test, True)

# Exemple 2 : The AND function
# -------------------------------------------------------------------------
print("\n\n===============================")
print("learning the AND function")
print("===============================")

train_input = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_output = numpy.array([[0], [0], [0], [1]])
train = {'input': train_input, 'output': train_output}

test = train.copy()

network.reset()

print("Performance before learning")
network.test_regression(test, True)

network.learn(train, 50000, 0.1, False)

print("\nPerformance after  learning")
network.test_regression(test, True)

# Exemple 3 : The XOR function
# -------------------------------------------------------------------------
print("\n\n================================")
print("learning the XOR function")
print("================================")

train_input = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_output = numpy.array([[0], [1], [1], [0]])
train = {'input': train_input, 'output': train_output}

test = train.copy()

network.reset()
print("Performance before learning")
network.test_regression(test, True)

network.learn(train, 50000, 0.1, False)

print("\nPerformance after learning")
network.test_regression(test, True)
