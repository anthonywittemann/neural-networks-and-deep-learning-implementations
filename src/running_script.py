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
net = network.Network([784, 30, 10])

# run stochastic gradient descent to train the net
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)