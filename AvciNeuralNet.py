
# coding: utf-8

# # Simple feedforward neural network with Stochastic Gradient Descent learning algorithm
# 
# Tamer Avci, June 2016
# 
# Motivation: To get better insights into Neural nets by implementing them only using NumPy.  
# Task: To classify MNIST handwritten digits: http://yann.lecun.com/exdb/mnist/
# 
# Accuracy:~95%
# 
# Deficiencies:
# No optimization of the code
# No cross-validation split.
# 

# 1) Define a network class. Instance variables will be:
#      -num of layers e.g. [2,3,1]
#      -size of our network [2, 3, 1] = 3
#      -bias vectors
#      -weight matrices
#      
# 2) Write the instance methods:
#     -feedforward: used to pass the input forward once and evaluate
#     -train: take batches of the data and call SGD to minimize the cost function
#     -SGD: for every single input in the batch, call backprop to find the derivatives with respect to
#           weights and bias, and then update those in the descent direction
#     -backprop: find the derivatives using chain rule (watch the video for more help)
#     -evaluate: run feedforward after training and compare it against target value
#     -cost_derivative: as simple as (y(target) - x(input))
#     -sigmoid: sigmoid function (1 / 1 - exp(-z))
#     -sigmoid delta: derivative of the sigmoid function

# In[ ]:

#import dependencies
import random
import numpy as np


# In[46]:

class Network(object):

    def __init__(self, layer): 
        
        # layer: input layer array
        # Example usage >>> net = Network([2,3,1])
        # For instance [2,3,1] would imply 2 input neurons, 3 hidden neurons and 1 output neuron
        
        self.num_layers = len(layer)
        self.sizes = layer
        
        # Create random bias vectors for respective layers using uniform distribution
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        
        # Create weight matrices for respective layers, note the swap x,y
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        

    def feedforward(self, a):
        
        """ For each layer take the bias and the weight, and apply sigmoid function 
        Return the output of the network if "a" is input. Used after training when evaluating"""
        
        for b, w in zip(self.biases, self.weights):
            #One complete forward pass
            a = sigmoid(np.dot(w, a)+b)
        return a

    def train(self, training_data, epochs, mini_batch_size, eta, test_data=test_data):
        
        """ Train the neural network taking batches from the data hence stochastic SGD """
        
        #provide training data, number of epochs to train, batch size, the learning rate and the test_data
        #learning rate: how fast do you want the SGD to proceed
        
        n_test = len(test_data)
        n = len(training_data)
        
        for j in xrange(epochs):
            
            random.shuffle(training_data) #shuffle the data
            
            mini_batches = [training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.SGD(mini_batch, eta)

        #uncomment this if you want intermediate results
		#print "Batch: {0} / {1}".format(self.evaluate(test_data), n_test) 
        
        #otherwise epoch results:
            print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)

    def SGD(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation (derivatives) to a single mini batch."""
        
        #provide the mini_batch and the learning rate
        
        #create empty derivative array of vectors for each layer
        nabla_b = [np.zeros(b.shape) for b in self.biases] 
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        
        for x, y in mini_batch:
            #call backprop for this particular input
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #update the derivative
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        #update the weight and biases in the negative descent direction aka negative derivative    
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        
        """For in-depth computation of the derivatives watch my video!"""
        
        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        
        # backward pass <<<--- output layer
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        #delta is simply cost_derivate times the sigmoid function of the last activation aka output which
        #is the derivative of the bias for output layer
        delta_nabla_b[-1] = delta
        #derivative of the weights:
        #delta * output of the previous layer which is the previous activation before the output
        delta_nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        #backward pass <<<--- hidden layers
        #using negative indices: starting from the first hidden layer and backwards
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            delta_nabla_b[-l] = delta
            delta_nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            
        return (delta_nabla_b, delta_nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)


    def sigmoid(z):
        """ Classic sigmoid function: lot smoother than step function
        that MLP uses because we want small changes in parameters lead to
        small changes in the output """
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return sigmoid(z)*(1-sigmoid(z))


# Sources:
# Neural networks and deep learning, Michael Nielsen (great book!)
# http://neuralnetworksanddeeplearning.com/chap2.html
# Neural network, backpropagation algorithm, Ryan Harris
# https://www.youtube.com/watch?v=zpykfC4VnpM
# MNIST database
# 
# Code:
# mnist_loader is completely Nielsen's work
# Network code is also majorly inspired by him. I have made modifications to make it simpler.
