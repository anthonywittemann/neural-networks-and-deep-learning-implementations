"""
running_script.py
~~~~~~~~~~

This is a script that can be used instead of the 
command line to automate loading data, creating the net, 
and start training it on the data using SGD
"""

import mnist_loader
import network

# load the data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# initialize the net with 784 input nodes, 30 hidden nodes (can vary), 10 output nodes
net = network.Network([784, 30, 10]) # ~ 95% max accuracy
# net = network.Network([784, 100, 10]) # 100 nodes in hidden layer ~ 68% accuracy
# net = network.Network([784, 10]) # 2 layers, no hidden layer ~ TODO try ~ 69% accuracy
# net = network.Network([784, 30, 30, 10]) # extra hidden layer, also ~ 95% max accuracy

# run stochastic gradient descent to train the net
# train for 30 epochs, mini-batch size of 10, learning rate, alpha, of 3.0
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
# changing batch size and learning rate - 93.6% accuracy
#net.SGD(training_data, 30, 20, 1.0, test_data=test_data)
# learning rate 5.0
net.SGD(training_data, 30, 20, 5.0, test_data=test_data)