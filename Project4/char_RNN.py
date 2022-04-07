# Project 4: Text Generation with RNNs
# Author: Duong Hoang
# CS 460G - 001
# Due Date: Apr 11th, 2022

'''
    Purpose: predict the handwritten number using the pixel of a 28x28 image
    Pre-cond: a mnist training data csv file and a mnist testing data csv file
    Post-cond: multilayer perceptron model, predicted handwritten number, 
                and accuracy of multiclass classifier

'''

### Implementation ###

# initialize
DATA_FILE = 'tiny-shakespeare.txt'
ALPHA = 0.1                     # learning rate
NUM_EPOCHS = 150
NUM_HIDDEN_LAYER_NODES = 100
NUM_HIDDEN_LAYERS = 1   

# import libraries
import numpy as np
import torch
from torch import device, nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# initialize


def create_one_hot(sequence, vocab_size):
    #Tensor is of the form (batch size, sequence length, one-hot length)
    encoding = np.zeros((1, len(sequence), vocab_size), dtype=np.float32)
    
    for i in range(len(sequence)):
        encoding[0, i, sequence[i]] = 1
        
    return encoding
    

class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the network!
		# Batch first defines where the batch parameter is in the tensor
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        # Fully connected layers
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        hidden_state = self.init_hidden()

        output, hidden_state = self.rnn(x, hidden_state)
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.fc(output)
        
        return output, hidden_state
        
    def init_hidden(self):
        #Hey,this is our hidden state. Hopefully if we don't have a batch it won't yell at us
        #Also a note, pytorch, by default, wants the batch index to be the middle dimension here. 
        #So it looks like (row, BATCH, column)
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden

class CharRNN(RNNModel):
    def __init__(self, sentences, hidden_size, num_layers, learning_rate, num_epochs):
        self.sentences = sentences
        self.intChar = dict(enumerate(set(''.join(sentences))))
        self.charInt = {character: index for index, character in self.intChar.items()}
        self.vocab_size = len(self.charInt)
        self.model = self.train(sentences, hidden_size, num_layers, learning_rate, num_epochs)

    def create_sequences(self, sentences):
        # offset input and output sentences
        input_sequence = []
        target_sequence = []
        for i in range(len(sentences)):
            # remove the last character from the input sequence
            input_sequence.append(sentences[i][:-1])
            # remove the first element from target sequences
            target_sequence.append(sentences[i][1:])

        # construct the one hots! First step, replace all characters with integer
        for i in range(len(sentences)):
            input_sequence[i] = [self.charInt[character] for character in input_sequence[i]]
            target_sequence[i] = [self.charInt[character] for character in target_sequence[i]]
        
        return input_sequence, target_sequence

    def get_device(self):
        # check the device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is available")
        else:
            device = torch.device("cpu")
            print("GPU not available, CPU used")
        
        return device

    def train(self, sentences, hidden_size, num_layers, learning_rate, num_epochs):
        # get sequences
        input_sequence, target_sequence = self.create_sequences(sentences)
        # get device
        device = self.get_device()
        # create RNN model
        model = RNNModel(self.vocab_size, self.vocab_size, hidden_size, num_layers)
        # define loss
        loss = nn.CrossEntropyLoss()
        # use Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for i in range(len(input_sequence)):
                optimizer.zero_grad()
                x = torch.from_numpy(create_one_hot(input_sequence[i], self.vocab_size))
                x.to(device)
                y = torch.Tensor(target_sequence[i])
                output, hidden = model(x)

                loss_val = loss(output, y.view(-1).long())
                loss_val.backward() # backpropagation
                optimizer.step()

        print("Loss: {:.4f}".format(loss_val.item()))
        
        return model

    def predict(self, model, character):
        
        characterInput = np.array([self.charInt[c] for c in character])
        characterInput = create_one_hot(characterInput, self.vocab_size)
        characterInput = torch.from_numpy(characterInput)
        out, hidden = model(characterInput)
        
        #Get output probabilities
        prob = nn.functional.softmax(out[-1], dim=0).data
        
        character_index = torch.max(prob, dim=0)[1].item()
        
        return self.intChar[character_index], hidden
    

def main():
    # read txt file
    file = open(DATA_FILE, 'r')
    sentences = file.readlines()
    file.close()

    # create char-RNN model
    model = CharRNN(sentences, NUM_HIDDEN_LAYER_NODES, NUM_HIDDEN_LAYERS, NUM_EPOCHS)

    
main()