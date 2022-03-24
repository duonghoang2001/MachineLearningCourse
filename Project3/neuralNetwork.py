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
NUM_OUTPUT_NODES = 1            # binary
NUM_HIDDEN_LAYER_NODES = 5

# import libraries
import pandas as pd
import numpy as np

class NeuralNetwork():
    def __init__(self, X: pd.DataFrame, Y: pd.DataFrame, num_output_nodes: int, 
                num_hidden_layer_nodes: int, alpha: float):
        self.norm_X = self.normalize_data(X.to_numpy())
        self.Y = Y.to_numpy()
        self.num_output_nodes = num_output_nodes
        self.num_hidden_layer_nodes = num_hidden_layer_nodes
        self.weights_h, self.weights_o, self.biases_h, self.biases_o = self.multilayer_perceptron(
                                            self.norm_X, self.Y, alpha)


    def activation_function(self, x): 
        '''Return activation function: g = sigmoid'''

        return 1 / (1 + np.exp(-x))


    def deriv_activation_function(self, x):
        '''Return derivative of activation function: g' = g * (1-g)'''

        return self.activation_function(x) * (1 - self.activation_function(x))

            
    def normalize_data(self, data: np.matrix):
        '''Return normalize feature values to range 0-1'''

        return data / 255


    def multilayer_perceptron(self, X: np.matrix, Y: np.matrix, alpha: float):
        '''Return a vector of thetas using stochastic gradient descent'''

        weights_h = np.random.uniform(-1, 1, (X.shape[1], self.num_hidden_layer_nodes)) # [784 x 5]
        weights_o = np.random.uniform(-1, 1, (self.num_hidden_layer_nodes, self.num_output_nodes)) # [5 x 1]
        biases_h = np.random.uniform(-1, 1, (self.num_hidden_layer_nodes, 1)) # [5 x 1]
        biases_o = np.random.uniform(-1, 1, (self.num_output_nodes, 1)) # [1 x 1] 
        
        for i in range(len(X)):
            example = X[i][:, np.newaxis]
            class_label = Y[i][:, np.newaxis]
            ### forward pass 
            # generate hidden outputs
            # ([5 x 784] x [784 x 1] = [5 x 1]) + [5 x 1] = [5 x 1]
            hidden_nodes = np.dot(weights_h.transpose(), example) + biases_h
            activations_h = list(map(self.activation_function, hidden_nodes))
            # generate the full output
            # ([1 x 5] x [5 x 1] = [1 x 1]) + [1 x 1] = [1 x 1]
            activations_o = self.activation_function(
                        np.dot(weights_o.transpose(), activations_h) + biases_o)
                
            ### backpropagation
            # calculate deltas
            # ([1 x 1] - [1 x 1] = [1 x 1]) * [1 x 1] = [1 x 1]
            delta_o = (class_label - activations_o) * self.deriv_activation_function(class_label)
            # ([5 x 1] * [1 x 1] = [5 x 1]) * [5 x 1] = [5 x 1]
            delta_h = np.matmul(weights_o, delta_o) * self.deriv_activation_function(hidden_nodes)
        
            # propagate deltas backward
            # [784 x 1] x [1 x 5] = [784 x 5]
            gradients_h = np.outer(example, np.transpose(delta_h))
            # update hidden weights and biases
            weights_h += alpha * gradients_h
            biases_h += alpha * delta_h
            # update output weights and biases
            # [5 x 1] * [1 X 1]
            weights_o += alpha * np.matmul(hidden_nodes, delta_o)
            biases_o += alpha * delta_o
        
        return weights_h, weights_o, biases_h, biases_o


    def predict(self, X: np.matrix):
        '''Return predictions of multilayer perceptron model'''

        # get model predictions
        predictions = []
        for i in range(len(X)):
            example = X[i][:, np.newaxis]
            # generate hidden outputs
            # ([5 x 784] x [784 x 1] = [5 x 1]) + [5 x 1] = [5 x 1]
            hidden_nodes = np.dot(self.weights_h.transpose(), example) + self.biases_h
            activations_h = list(map(self.activation_function, hidden_nodes))
            # generate the full output
            # ([1 x 5] x [5 x 1] = [1 x 1]) + [1 x 1] = [1 x 1]
            activations_o = self.activation_function(
                np.dot(self.weights_o.transpose(), activations_h) + self.biases_o)
            
            # classify example
            if activations_o > 0.5: predictions.append(1)
            else: predictions.append(0)

        return predictions


    def calculate_accuracy(self, X: pd.DataFrame, Y: pd.DataFrame):
        '''Return accuracy rate of predictions made by neural network'''

        # normalize examples
        norm_X = self.normalize_data(X.to_numpy())

        # get model predictions
        predictions = self.predict(norm_X)

        # create csv file with first column is prediction, second column is key
        nn_predictions = pd.DataFrame(predictions, columns=['predict'])
        compare = pd.concat([nn_predictions, Y], axis=1, join='inner')
        compare.to_csv(path_or_buf=f'classified_{TESTING_FILE}.csv', index=False)

        correct_ct = 0 # count correct predictions
        # compare prediction with test key
        Y = Y.to_numpy()
        for i in range (len(Y)):
            if predictions[i] == Y[i]: correct_ct += 1

        return correct_ct/len(Y)


def main():
    # read training data
    train_data = pd.read_csv(f'{TRAINING_FILE}.csv')   
    train_X = pd.DataFrame(train_data.iloc[:, 1:])    # features
    train_Y = pd.DataFrame(train_data.iloc[:, 0])     # class label

    # read testing data
    test_data = pd.read_csv(f'{TESTING_FILE}.csv')   
    test_X = pd.DataFrame(test_data.iloc[:, 1:])    # features
    test_Y = pd.DataFrame(test_data.iloc[:, 0])     # class label

    model = NeuralNetwork(train_X, train_Y, NUM_OUTPUT_NODES, NUM_HIDDEN_LAYER_NODES, ALPHA)
    print('Accuracy =', model.calculate_accuracy(test_X, test_Y))

main()