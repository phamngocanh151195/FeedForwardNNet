import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#check the directory for mnist archive in mnist_loader.py

import AvciNeuralNet
net = AvciNeuralNet.Network([784, 30,10])

net.train(training_data, 30, 10, 3.0, test_data=test_data)
