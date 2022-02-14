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
DATA_FILE = 'synthetic-4.csv'
INVALID_VALUE = -1

# import libraries
import pandas as pd                     # analysing data
import numpy as np                      # handling data
import matplotlib.pyplot as plt         # visualizing data


class Node():
    def __init__(self, value=None):
        self.value = value
        self.children = []

    
    def is_leaf(self):
        return len(self.children) == 0
  

class DecisionTreeClassifier(Node):
    def __init__(self, data):
        self.label_index = len(data.columns) - 1
        self.features_list = (data.columns.tolist())[:self.label_index]
        self.boundaries = [self.find_boundaries(data.iloc[:, feature]) for feature in self.features_list]
        self.discretized_data = self.discretize_training_data(data)

        self.root = self.ID3(self.discretized_data, self.label_index, self.features_list, 0)

    def get_entropy(self, class_label_data: pd.DataFrame):
        '''Return entropy = sum of each class label probability times its log2'''

        data_ct = class_label_data.count()

        # get a list of each class label probabilities
        probs = [value_ct/data_ct for value_ct in class_label_data.value_counts()]
            
        return sum([-prob * np.log2(prob) for prob in probs])
    

    def get_info_gain(self, data:pd.DataFrame, feature_ind, label_ind):
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
            feature_entropy += value_prob * self.get_entropy(sub_feature_data.iloc[:, label_ind])
    
        return self.get_entropy(data.iloc[:, label_ind]) - feature_entropy


    def ID3(self, examples: pd.DataFrame, target_attr, attrs: list, depth):
        '''Implementation of ID3 algorithm to build decision tree.
        Return root node of decision tree
        '''

        root = Node() # root node for decision tree
        
        # get list of examples' unique class labels
        unique_class_labels = examples.iloc[:, target_attr].value_counts().index.to_list()

        # if examples are homogeneous, return root with label = unique val of targetAttr
        if len(unique_class_labels) == 1: 
            root.value = unique_class_labels[0]
        # if attrs is empty, return root with label = most common val of targetAttr
        elif len(attrs) == 0: 
            root.value = unique_class_labels[0]
        # otherwise begin
        else:
            # A <- the attr from attrs that best classifies examples
            gains = [self.get_info_gain(examples, attr, target_attr) for attr in attrs]
            A = attrs[gains.index(max(gains))]
            # the decision attr for root <- A
            root.value = A
            # for each possible value, vi, of A,
            unique_values = examples.iloc[:, A].unique()
            for value in unique_values:
                # add a new tree branch below root, corresponding to the test A = vi
                new_branch = Node(value)
                root.children.append(new_branch)
                # let examples(vi) be the subset of examples that have vi for A
                branch_examples = examples.loc[examples.iloc[:, A] == value]
                # if examples(vi) is empty,
                if len(branch_examples) == 0 or depth == MAX_DEPTH - 1:
                    # then below this branch add a leaf node with 
                    # label = most common val of targetAttr in examples
                    leaf = Node()
                    leaf.value = examples.iloc[:, target_attr].value_counts().idxmax()
                    new_branch.children.append(leaf)
                else:
                    # below this new branch add the subtree
                    # make a new copy of attrs without A
                    new_attrs = [attr for attr in attrs if attr != A] 
                    child = self.ID3(branch_examples, target_attr, new_attrs, depth + 1)
                    new_branch.children.append(child)

        return root


    def printTree(self, root: Node, depth):
        for i in range(depth):
            print("\t", end="")
        print(root.value, end="")
        if root.is_leaf():
            print(" -> ", root.value)
        print()
        for child in root.children:
            self.printTree(child, depth + 1)


    def find_boundaries(self, data: pd.DataFrame):
        '''Return boundaries that separate data to equal distance intervals'''

        # divide data into NUM_BINS feature intervals
        intervals = data.value_counts(bins=NUM_BINS).sort_index(ascending=True)
        # find boundaries which are the right ends of the feature intervals
        boundaries = [intervals.index[i].right for i in range(NUM_BINS - 1)]

        return boundaries


    def get_discretize_value(self, value, boundaries: list):
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


    def discretize_features(self, data: pd.DataFrame):
        '''Return equal distance discretized dataset of given continuous dataset'''

        # initialize new dataset and list of boundaries
        discretized_data = pd.DataFrame()

        num_features = len(data.columns)
        # equal distance discretize each feature
        for col in range(num_features):
            # discretize feature values based on the boundaries
            discretized_feature = []
            for i in range(len(data)): # for each feature value
                discretized_feature.append(self.get_discretize_value(data.at[i, col], self.boundaries[col]))
            
            # add new discretized feature values to new dataset
            discretized_data[col] = discretized_feature

        #discretized_data.to_csv(path_or_buf=f'disc_{DATA_FILE}', index=False, header=None)
        
        return discretized_data


    def discretize_training_data(self, data: pd.DataFrame):
        '''Return equal distance discretized training dataset of 
        given continuous training dataset with class labels
        '''
        # discretize all features
        discretized_data = self.discretize_features(data.iloc[:, :self.label_index])

        # add class_label column (last column) to discretized data
        discretized_data[self.label_index] = data.iloc[:, self.label_index] 

        return discretized_data


    def classify_datum(self, current: Node(), datum: pd.DataFrame):
        '''Return classification prediction of datum using decision tree'''
        classification = INVALID_VALUE # initialize prediction

        if not(current.is_leaf()): # if current node is not leaf
            # compare datum feature value to each branch condition
            child_num = 0
            while not(datum[current.value] == current.children[child_num].value):
                child_num += 1

            branch = current.children[child_num] # branch that macth feature value

            if branch.children[0].is_leaf(): # if branch node is leaf, assign leaf value
                classification = branch.children[0].value
            else: # if not, check the next interior node
                classification = self.classify_datum(branch.children[0], datum)

        else: # if current node is leaf, assign predict value
            classification = current.value
        
        return classification



    def classify_data(self, test_data: pd.DataFrame):
        '''Return list of test data classification predictions'''

        discretized_test_data = self.discretize_features(test_data).values.tolist()
        
        classifications = [] # initialize list of predictions for each datum

        for i in range(len(discretized_test_data)):
            classifications.append(self.classify_datum(self.root, discretized_test_data[i]))
        
        return classifications


    def calculate_accuracy(self, test_data: pd.DataFrame, test_label_key: pd.DataFrame):
        '''Return accuracy rate of decision tree predictions'''

        # compare datasets as lists of labels
        predictions = self.classify_data(test_data) 
        test_key = test_label_key.values.tolist()

        compare = pd.DataFrame(list(zip(predictions, test_key)), columns=['Prediction', 'Key'])
        compare.to_csv(path_or_buf=f'comparision.csv', index=False)

        correct_ct = 0 # count correct predictions
        data_ct = len(test_key) # total number of data to compare
        # compare prediction with test key
        for i in range (data_ct):
            #print(predictions[i, 0])
            if predictions[i] == test_key[i]: correct_ct += 1

        return correct_ct / data_ct


def main():
    # read data
    data = pd.read_csv(DATA_FILE, header=None)   
    label_index = len(data.columns) - 1
    test_data = data.iloc[:, :label_index]
    test_label_key = data.iloc[:, label_index]

    # make decision tree
    tree = DecisionTreeClassifier(data)
    tree.printTree(tree.root, 0)

    # calculate predictions accuracy
    print('Accuracy of tree =', tree.calculate_accuracy(test_data, test_label_key))    

main()