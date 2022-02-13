# Project 1: Decision Trees
# Author: Duong Hoang
# CS 460G - 001
# Due Date: Feb 15th, 2022

'''
    Purpose: predict class label of a given data using decision tree
    Pre-cond: a synthetic data csv file
    Post-cond: decision tree, predicted class label, and accuracy of predictions

'''

### Implementation ###

# initialize
MAX_DEPTH = 3
NUM_BINS = 4 # always greater than or equal to 1
DATA_FILE = 'synthetic-2.csv'
INVALID_VALUE = -1

# import libraries
import re
import pandas as pd             # analysing data
import numpy as np              # handling data
import matplotlib.pyplot as plt # visualizing data
 
class Node():
    def __init__(self, value=None):
        self.value = value
        self.children = []
    
    def is_leaf(self):
        return len(self.children) == 0
      
        
def get_entropy(class_label_data: pd.DataFrame):
    '''Return entropy = sum of each class label probability times its log2'''

    data_ct = class_label_data.count()

    # get a list of each class label probabilities
    probs = [value_ct/data_ct for value_ct in class_label_data.value_counts()]
        
    return sum([-prob * np.log2(prob) for prob in probs])
  

def get_info_gain(data:pd.DataFrame, feature_ind, label_ind):
    '''Return info gain for feature which is the difference of 
    the entropy before and after spliting on feature     
    '''
    
    feature_entropy = 0 # initialize feature entropy
    data_ct = len(data)

    # get list of unique feature values in value name ascending order
    unique_values = np.unique(data.iloc[:, feature_ind])

    # calculate each unique value's entropy
    for value in unique_values:
        # create a sub-dataset of rows that contain unique value
        sub_feature_data = data.loc[data.iloc[:, feature_ind] == value]
        # add each unique value's entropy to feature entropy
        value_prob = len(sub_feature_data) / data_ct
        feature_entropy += value_prob * get_entropy(sub_feature_data.iloc[:, label_ind])
    
    return get_entropy(data.iloc[:, label_ind]) - feature_entropy


def ID3(examples: pd.DataFrame, target_attr, attrs: list):
    '''Implementation of ID3 algorithm to build decision tree.
    Return root node of decision tree
    '''

    root = Node() # root node for decision tree
    
    # get list of examples' unique class labels
    unique_class_labels = examples.iloc[:, target_attr].value_counts()

    # if examples are homogeneous, return root with label = unique class label
    if len(unique_class_labels) == 1: 
        root.value = unique_class_labels.index.to_list()[0]
    # if attrs is empty, return root with label = most common val of targetAttr
    elif len(attrs) == 0: 
        root.value = unique_class_labels.idxmax()
    # otherwise begin
    else:
        # A <- the attr from attrs that best classifies examples
        gains = [get_info_gain(examples, attr, target_attr) for attr in attrs]
        A = attrs[gains.index(max(gains))]

        # the decision attr for root <- A
        # for each possible value, vi, of A,
        unique_values = examples.iloc[:, A].unique()
        for value in unique_values:
            # add a new tree branch below root, corresponding to the test A = vi
            new_branch = Node(value)
            root.children.append(new_branch)
            # let examples(vi) be the subset of examples that have vi for A
            branch_examples = examples.loc[examples.iloc[:, A] == value]
            # if examples(vi) is empty,
            if len(branch_examples) == 0:
                # then below this branch add a leaf node with 
                # label = most common val of targetAttr in examples
                leaf = Node(branch_examples.iloc[:, target_attr].value_counts().idxmax())
                new_branch.children.append(leaf)
            else:
                # below this new branch add the subtree
                # make a new copy of attrs without A
                new_attrs = [attr for attr in attrs if attr != A] 
                child = ID3(branch_examples, target_attr, new_attrs)
                new_branch.children.append(child)

    return root

'''
def print_tree(root: Node):

    if root.is_leaf:
        print(root.value)
    else:
        print('{')
        for child in root.children:
            print(f'\t{child}: ', end='')
            print_tree(child)
        print('}')
'''
def printTree(root: Node, depth):
    for i in range(depth):
        print("\t", end="")
    print(root.value, end="")
    if root.is_leaf():
        print(" -> ", root.value)
    print()
    for child in root.children:
        printTree(child, depth + 1)

def find_boundaries(data: pd.DataFrame):
    '''Return boundaries that separate data to equal distance intervals'''

    # divide data into NUM_BINS feature intervals
    intervals = data.value_counts(bins=NUM_BINS).sort_index(ascending=True)
    # find boundaries which are the right ends of the feature intervals
    boundaries = [intervals.index[i].right for i in range(NUM_BINS - 1)]

    return boundaries


def get_discretize_value(value, boundaries: list):
    '''Return discretized value = the interval number that the value belongs to'''

    discretized_value = INVALID_VALUE

    # loop through each interval except the last one [last bound, inf)
    for j in range(NUM_BINS - 1): # len(boundaries) == NUM_BINS - 1
        if value < boundaries[j]: # if belonged to interval
            discretized_value = j # update discretized value
            break 
    # if fails to categorize in the previous intervals,
    # value is in interval [last bound, inf)
    if discretized_value == -1: discretized_value = NUM_BINS - 1

    return discretized_value


def discretize_data(data: pd.DataFrame):
    '''Return equal distance discretized dataset of given continuous dataset'''

    # initialize new dataset
    discretized_data = pd.DataFrame()

    num_features = len(data.columns) - 1
    # equal distance discretize each feature
    for col in range(num_features):
        boundaries = find_boundaries(data.iloc[:, col])

        # discretize feature values based on the boundaries
        discretized_feature = []
        for i in range(len(data)): # for each feature value
            discretized_feature.append(get_discretize_value(data.at[i, col], boundaries))
        
        # add new discretized feature values to new dataset
        discretized_data[col] = discretized_feature
    
    # duplicate class_label column (last column) to discretized data
    discretized_data[num_features] = data.iloc[:, num_features] 

    #discretized_data.to_csv(path_or_buf=f'disc_{DATA_FILE}', index=False, header=None)
    
    return discretized_data


def main():
    # read data
    data = pd.read_csv(DATA_FILE, header=None)    
    
    disc_data = discretize_data(data)
    target_attr = len(disc_data.columns) - 1
    attrs = (disc_data.columns.tolist())[:target_attr]
    
    root = ID3(disc_data, target_attr, attrs)
    printTree(root, 0)
    

main()