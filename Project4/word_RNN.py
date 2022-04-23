# Project 4: Text Generation with RNNs
# Author: Duong Hoang
# CS 460G - 001
# Due Date: Apr 24th, 2022

'''
    Purpose: generate new text word by word
    Pre-cond: a text file
    Post-cond: RNN model, new sample text of given size

'''

### Implementation ###

# initialize
DATA_FILE = 'tiny-shakespeare.txt'  # text
ALPHA = 0.005                       # learning rate
NUM_EPOCHS = 10                     # iterations
BATCH_SIZE = 40                     # size of mini batch 
SEQUENCE_LEN = 20                   # sequence length
NUM_HIDDEN_LAYER_NODES = 800        # hidden nodes per layer
NUM_HIDDEN_LAYERS = 2               # hidden layers
DROPOUT_RATE =  0.5                 # dropout layer rate
TOP_K = 7                           # most n probable characters
OUTPUT_FILE = "generated_word.txt"  # sample output to file

# import libraries
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# check if GPU is available
has_cuda = torch.cuda.is_available()
# assign device
device = torch.device("cuda" if has_cuda else "cpu")


class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, drop_rate):
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Dropout layer
        self.dropout = nn.Dropout(drop_rate)
        # define the network
        # batch first defines where the batch parameter is in the tensor
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                    dropout=drop_rate, batch_first=True)
        # fully connected layers
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden_state):
        # get outputs and hidden state from rnn
        output, hidden_state = self.rnn(x, hidden_state)
        # pass output through dropout layer to avoid overfitting
        output = self.dropout(output)
        # stack up outputs
        output = output.contiguous().view(-1, self.hidden_size)
        # pass through fully-connected layer
        output = self.fc(output) 
        # output: [seq_len * batch_size, vocab_size]
        return output, hidden_state
        
    def init_hidden(self, batch_size):
        # initialize rnn states
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device), 
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))


def create_sequences(charInt: dict, data: str, batch_size: int, seq_len: int):
        # replace all characters with integer
        sentences = np.array([charInt[ch] for ch in data])
        # calculate total batch size that make up from sequences
        total_batch_size = batch_size * seq_len
        # total number of batches can be created
        num_batches = len(sentences) // total_batch_size
        # cut off left over characters that don't make a batch
        sentences = sentences[:num_batches * total_batch_size]
        # reshape into batch_size length rows
        sentences = sentences.reshape((batch_size, -1))
        
        # create mini batches
        for i in range(0, sentences.shape[1], seq_len):
            # offset input and output sentences
            # remove the last character from the input sequence
            input_sequence = sentences[:, i:i+seq_len]
            # remove the first element from target sequences
            target_sequence = np.zeros_like(input_sequence)
            try:
                target_sequence[:, :-1], target_sequence[:, -1] = input_sequence[:, 1:], sentences[:, i+seq_len]
            except IndexError:
                target_sequence[:, :-1], target_sequence[:, -1] = input_sequence[:, 1:], sentences[:, 0]
                
            yield input_sequence, target_sequence


def create_one_hot(sequence: np.array, vocab_size: int):
    # Tensor is of the form (batch size, sequence length, one-hot length)
    encoding = np.zeros((np.multiply(*sequence.shape), vocab_size), dtype=np.float32)
    encoding[np.arange(encoding.shape[0]), sequence.flatten()] = 1
    encoding =  encoding.reshape((*sequence.shape, vocab_size))
    
    return encoding
    

def train(model: RNNModel, data: str, charInt: dict, batch_size: int, seq_len: int, 
        num_epochs: int, learning_rate: float, clip=5):
    '''Train RNN model'''

    # get vocab size
    vocab_size = len(charInt)
    # define loss
    loss = nn.CrossEntropyLoss()
    # use Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # start training
    print("Start Training RNN model...")
    model.train()   
    if torch.cuda.is_available(): model.cuda()
    else: model.cpu()
    
    for i in range(num_epochs):
        # initialize hidden state
        hidden = model.init_hidden(batch_size)
        current_loss = 0

        for x, y in create_sequences(charInt, data, batch_size, seq_len):
            # encode batch into one-hot vector and make it Torch tensor
            x = create_one_hot(x, vocab_size)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            if has_cuda: inputs, targets = inputs.cuda(), targets.cuda()

            # otherwise, creating new variables for the hidden state
            hidden = tuple([var.data for var in hidden])

            # clear previous gradients
            optimizer.zero_grad()
            
            # get model's output 
            output, hidden = model(inputs, hidden)
            
            # calculate loss and perform backprop
            lossValue = loss(output, targets.view(batch_size * seq_len).long())
            lossValue.backward()

            # prevent the exploding gradient problem
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
            current_loss = lossValue.item()
        
        # report loss
        print(f"Epoch {i}: Loss = {current_loss:.4f}...") 


def predict(model, char, intChar: dict, charInt: dict, hidden=None, top_k=None):
        '''Return next character'''
        
        vocab_size = len(charInt)
        # Tensor inputs
        x = np.array([[charInt[char]]])
        x = create_one_hot(x, vocab_size)
        inputs = torch.from_numpy(x)
        
        if has_cuda: inputs = inputs.cuda()
        
        # detach hidden state from history
        hidden = tuple([var.data for var in hidden])
        # get the output of the model
        out, hidden =  model(inputs, hidden)

        # get the character probabilities
        prob = F.softmax(out, dim=1).data
        if has_cuda: prob = prob.cpu() 
        
        # get top characters
        if top_k is None:
            top_ch = np.arange(vocab_size)
        else:
            prob, top_ch = prob.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        # select the likely next character with some element of randomness
        prob = prob.numpy().squeeze()
        char = np.random.choice(top_ch, p=prob/prob.sum())
        
        # return the encoded value of the predicted char and the hidden state
        return intChar[char], hidden
        

def sample(model: RNNModel, intChar: dict, charInt: dict, size, prime, top_k=None):
    '''Return new text'''

    if has_cuda: model.cuda()
    else: model.cpu()
    
    # evaluate model
    model.eval() 
    
    # run through the prime characters
    chars = [ch for ch in prime]
    hidden = model.init_hidden(1)
    for ch in prime:
        char, hidden = predict(model, ch, intChar, charInt, hidden, top_k=top_k)

    chars.append(char)
    
    # pass in the previous character and get a new one
    for i in range(size):
        char, hidden = predict(model, chars[-1], intChar, charInt, hidden, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

def main():
    # read txt file
    file = open(DATA_FILE, 'r')
    data = file.readlines()
    sentences = []
    for line in data:
        for word in line.split():
            sentences.append(word)
            sentences.append(" ")
        sentences.append("\n")
    file.close()

    # create char-RNN model
    intChar = dict(enumerate(tuple(set(sentences))))
    charInt = {character: index for index, character in intChar.items()}
    vocab_size = len(charInt)  
    model = RNNModel(vocab_size, vocab_size, NUM_HIDDEN_LAYER_NODES, 
                    NUM_HIDDEN_LAYERS, DROPOUT_RATE)
    # training the model
    train(model, sentences, charInt, BATCH_SIZE, SEQUENCE_LEN, NUM_EPOCHS, ALPHA)

    # output text generation
    print("\n\nSAMPLE:\n---------------------------------------\n")
    # generate text
    text_sample = sample(model, intChar, charInt, 1000, 'QUEEN', top_k=7)
    print(text_sample)

    # save output to a file
    f = open(OUTPUT_FILE, "w")
    f.write(text_sample)
    f.close()
main()