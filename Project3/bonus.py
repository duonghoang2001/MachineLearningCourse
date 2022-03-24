# Project 3: Multilayer Perceptrons
# Author: Duong Hoang
# CS 460G - 001
# Due Date: Mar 25th, 2022

'''
    Purpose: predict the handwritten number using the pixel of a 28x28 image
    Pre-cond: a mnist training data csv file and a mnist testing data csv file
    Post-cond: decision tree, predicted handwritten number, 
                and accuracy of classifier

'''

### Implementation ###

# initialize
TRAINING_FILE = 'mnist_train_0_1'
TESTING_FILE = 'mnist_test_0_1'
ALPHA = 0.01                    # learning rate
NUM_EPOCHS = 3000               # iterations
NUM_OUTPUT_NODES = 1            # binary
NUM_HIDDEN_LAYERS = 2
NUM_HIDDEN_LAYER_NODES = 500



# import libraries
import pandas as pd
import numpy as np

class NeuralNetwork():
    def __init__(self, X: pd.DataFrame, Y: pd.DataFrame, num_output_nodes: int, 
            num_hidden_layers: int, num_hidden_layer_nodes: int, alpha: float, epochs: int):
        self.X = X.to_numpy()
        self.Y = Y.to_numpy()
        self.num_output_nodes = num_output_nodes
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_layer_nodes = num_hidden_layer_nodes
        self.hidden_weights, self.output_weights = self.multilayer_perceptron(alpha, epochs)

    def activation_function(self, x): 
        '''Return activation function: g = sigmoid'''
        return 1 / (1 + np.exp(-x))

    def deriv_activation_function(self, g_x):
        '''Return derivative of activation function: g' = g * (1-g)'''
        return g_x * (1 - g_x)

    def initialize_weights(self):
        '''Return weights of all layers initialized randomly between 0 and 1'''
        weights = []
        w_ItoH = np.random.uniform(-1, 1, (self.X.shape[1], self.num_hidden_layer_nodes))
        weights.append(w_ItoH)
        for layer in range(self.num_hidden_layers - 1):
            weight = np.random.uniform(-1, 1, 
                    (self.num_hidden_layer_nodes, self.num_hidden_layer_nodes))
            weights.append(weight)

        return weights


    def forward_pass(self, example: np.matrix, hidden_weights: list, output_weights: np.matrix):
        '''Return activation at output layer using forward pass'''
        
        hidden_activations = [] # [500 x 1] * 2

        # calculate activations at hidden layers
        # [500 x 784] x [784 x 1] = [500 x 1]
        hidden_activations.append(self.activation_function(
                    np.dot(hidden_weights[0].tranpose(), example.transpose())))
        for layer in range(1, self.num_hidden_layers):
            # [500 x 500] x [500 x 1] = [500 x 1]
            hidden_activations.append(self.activation_function( 
                    np.dot(hidden_weights[layer], hidden_activations[layer - 1])))
        # calculate activations at output layer
        # transpose([500 x 1]) x [500 x 1] = [1 x 500] x [500 x 1] = [1 x 1]
        output_activation = self.activation_function(np.dot(output_weights.transpose(), 
                                    hidden_activations[self.num_hidden_layers - 1]))
        return hidden_activations, output_activation


    def backpropagation(self, hidden_weights: list, output_weights: np.matrix,
                         hidden_activations: list, output_activation: np.matrix):
        hidden_deltas = []
        # [500 x 1] x [500 x 1]
        delta_HtoO = self.deriv_activation_function(hidden_activations[len(hidden_activations) - 1]) * np.dot() output_weights
        for layer in range(self.num_hidden_layers - 1, -1, -1):

        return hidden_deltas


    def multilayer_perceptron(self, alpha: float, epochs: int):
        '''Return a vector of thetas using stochastic gradient descent'''

        hidden_weights = self.initialize_weights() # [[784 x 500], [500 x 500] * 2]
        output_weights = np.random.uniform(-1, 1, (self.num_hidden_layer_nodes, self.num_output_nodes)) # [500 x 1]

        for epoch in range(epochs):
            for example_index in range(self.X.shape[0]):
                # forward pass
                hidden_activations, output_activation = self.forward_pass(self.X[example_index], hidden_weights, output_weights)
                # calculate output error
                error = self.Y[example_index] - output_activation
                # calculate delta at the output layer
                delta_O = error * self.deriv_activation_function(error)
                # backpropagation
                deltas = self.backpropagation(example_index)

        return hidden_weights, output_weights



    def linear_regression(self, X: pd.DataFrame):
        '''Return predictions of linear regression model'''

        # normalize data
        norm_X = self.normalize_data(X).to_numpy()
        # add the bias feature x_0 = 1 to examples
        norm_X = np.insert(norm_X, 0, [1] * norm_X.shape[0], axis=1) 
        # get model predictions h_theta = theta_n * x_n
        predictions = norm_X.dot(self.thetas) 

        return predictions

def main():
    # read training data
    data = pd.read_csv(f'{TRAINING_FILE}.csv')   
    X = pd.DataFrame(data.iloc[:, 1:])    # features
    Y = pd.DataFrame(data.iloc[:, 0])     # classlabel

    model = NeuralNetwork(X, Y, NUM_OUTPUT_NODES, NUM_HIDDEN_LAYERS, 
                        NUM_HIDDEN_LAYER_NODES)

main()