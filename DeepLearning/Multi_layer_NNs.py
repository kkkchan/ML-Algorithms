import numpy as np
import matplotlib.pyplot as plt
from activations import relu, sigmoid, sigmoid_backward, relu_backward


class NN(object):
    """
    Implementation of multi-layer neural networks.

    Input shape of fit data: [num_of_samples, ...]
    

    """
    def __init__(self, layer_dims):
        '''
        Layer_dims is a array(or list, eta) data, which 
        includes dims of every neural networks, starting 
        from the input layer to the output layer.

        Parameters initialization is finished here.

        For example:
            [3,3,5,1] denotes a network with 2 hidden 
            layers, dim of 3, 5 respectively, and the 
            dim of input and output layer are 3, 1 
            respectively.


        '''
        self.parameters = {}
        self.L = len(layer_dims) - 1
        self.caches = []
        self.loss = []


        for l in range(1, self.L+1):
            self.parameters['W' + str(l)] = np.random.randn(layer_dims[l], 
                layer_dims[l - 1])
            self.parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

            assert self.parameters['W' + str(l)].shape == (layer_dims[l], 
                layer_dims[l - 1])
            assert self.parameters['b' + str(l)].shape == (layer_dims[l], 1)    


    def linear_forward(self, A, W, b):
        '''
        supports activation relu and sigmoid
        '''
        Z = np.dot(W, A) + b

        assert Z.shape == (W.shape[0], A.shape[1])
        cache = (A, W, b)

        return Z, cache


    def linear_activation_forward(self, A, W, b, activation):
        Z, linear_cache = self.linear_forward(A, W, b)
        if activation == 'sigmoid':
            A, activation_cache = sigmoid(Z), (linear_cache, Z)
        elif activation == 'relu':
            A, activation_cache = relu(Z), (linear_cache, Z)
        else:
            assert False
        assert A.shape == (W.shape[0], Z.shape[1])  

        return A, activation_cache


    def L_model_forward(self, X):
        self.caches = []
        A = X

        for l in range(1, self.L):
            A, cache = self.linear_activation_forward(A, 
                self.parameters['W' + str(l)], 
                self.parameters['b' + str(l)], 
                'relu')
            self.caches.append(cache)

        AL, cache = self.linear_activation_forward(A,
                self.parameters['W' + str(self.L)],
                self.parameters['b' + str(self.L)],
                'sigmoid')
        self.caches.append(cache)

        assert AL.shape == (1, X.shape[1])

        return AL


    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = - 1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

        cost = np.squeeze(cost)
        assert cost.shape == ()

        return cost


    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert dA_prev.shape == A_prev.shape
        assert dW.shape == W.shape
        assert db.shape == b.shape

        return dA_prev, dW, db


    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache

        if activation == 'relu':
            dZ = relu_backward(dA, activation_cache)
        elif activation == 'sigmoid':
            dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db


    def L_model_backward(self, AL, Y, caches):
        grads = {}
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = - (Y / AL - (1 - Y) / (1 - AL))

        current_cache = caches[self.L-1]
        grads['dA' + str(self.L-1)], grads['dW' + str(self.L)], grads['db' + str(self.L)] =\
            self.linear_activation_backward(dAL, current_cache, 'sigmoid')
        for l in reversed(range(self.L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads['dA' + str(l+1)],
                current_cache, 'relu')
            grads['dA' + str(l)] = dA_prev_temp
            grads['dW' + str(l+1)] = dW_temp
            grads['db' + str(l+1)] = db_temp

        return grads


    def update_parameters(self, parameters, grads, learning_rate):
        L = len(parameters) // 2
        for l in range(L):
            parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
        return parameters


    def fit(self, X, Y, learning_rate=0.1, max_iter=1000, print_cost=True):
        for i in range(max_iter):
            AL = self.L_model_forward(X.T)
            grads = self.L_model_backward(AL, Y, self.caches)
            self.parameters = self.update_parameters(self.parameters, grads, learning_rate)  
            
            cost = self.compute_cost(AL, Y.T)
            if (i+1) % 10 == 0:
                self.loss.append(cost)
                print('Current_loss:', cost)
        print('Model training finished!')


    def predict(self, X):
        A = self.L_model_forward(X.T)
        A = np.squeeze(A)
        Y_ = np.zeros_like(A)
        Y_[A > 0.5] = 1
        return Y_


    def score(self, X, Y):
        A = self.predict(X)
        return np.sum(A == Y) / X.shape[0]