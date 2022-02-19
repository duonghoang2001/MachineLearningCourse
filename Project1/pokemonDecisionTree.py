# Project 1: Decision Trees
# Author: Duong Hoang
# CS 460G - 001
# Due Date: Feb 18th, 2022

'''
    Purpose: predict legendary of a given pokemon data using decision tree
    Pre-cond: a pokemon stats data csv file and 
                its corresponding legendary classification data csv file
    Post-cond: decision tree, predicted legendary (True/False), 
                and accuracy ofclassifier

'''

### Implementation ###

# initialize
MAX_DEPTH = 3
NUM_BINS = 5 # always greater than or equal to 1
FEATURE_FILE = 'pokemonStats.csv'
LABEL_FILE = 'pokemonLegendary.csv'
INVALID_VALUE = -1

# import libraries
import pandas as pd                     # analysing data
import numpy as np                      # handling data


class Node():
    def __init__(self, value=None):
        self.value = value
        self.children = []

    
    def is_leaf(self):
        return len(self.children) == 0
  

class DecisionTreeClassifier(Node):
    def __init__(self, data, label, feature_names: list):
        self.label = label
        self.features = feature_names
        self.boundaries = {feature: self.find_boundaries(data[feature]) 
                            for feature in self.features}
        # discretize all non-binary features (features' whose names are not 'type')
        self.discretized_data = self.discretize_data(data, feature_names[:8]) 
        self.root = self.ID3(self.discretized_data, self.label, self.features, 0)


    def get_entropy(self, class_label_data: pd.DataFrame):
        '''Return entropy = sum of each class label probability times its log2'''

        data_ct = class_label_data.count()

        # get a list of each class label probabilities
        probs = [value_ct/data_ct for value_ct in class_label_data.value_counts()]
            
        return sum([-prob * np.log2(prob) for prob in probs])
    

    def get_info_gain(self, data:pd.DataFrame, feature, label):
        '''Return info gain for feature which is the difference of 
        the entropy before and after spliting on feature     
        '''
        
        feature_entropy = 0 # initialize feature entropy
        data_ct = len(data)

        # get list of unique feature values in value name ascending order
        unique_values = np.unique(data[feature])

        # calculate each unique value's entropy
        for value in unique_values:
            # create a sub-dataset of rows that contain unique value
            sub_feature_data = data.loc[data[feature] == value]
            # add each unique value's entropy to feature entropy
            value_prob = len(sub_feature_data) / data_ct
            feature_entropy += value_prob * self.get_entropy(sub_feature_data[label])
    
        return self.get_entropy(data[label]) - feature_entropy


    def ID3(self, examples: pd.DataFrame, target_attr, attrs: list, depth):
        '''Implementation of ID3 algorithm to build decision tree.
        Return root node of decision tree
        '''

        root = Node() # root node for decision tree
        
        # get list of examples' unique class labels
        unique_class_labels = examples[target_attr].value_counts().index.to_list()

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
            possible_value_range = NUM_BINS
            if 'Type' in A: possible_value_range = 2 # if A is binary feature
            for value in range(possible_value_range):
                # add a new tree branch below root, corresponding to the test A = vi
                new_branch = Node(value)
                root.children.append(new_branch)
                # let examples(vi) be the subset of examples that have vi for A
                branch_examples = examples.loc[examples[A] == value]
                # if examples(vi) is empty or tree about to reach max depth,
                if len(branch_examples) == 0 or depth == MAX_DEPTH - 1:
                    # then below this branch add a leaf node with 
                    # label = most common val of targetAttr in examples
                    leaf = Node()
                    leaf.value = examples[target_attr].value_counts().idxmax()
                    new_branch.children.append(leaf)
                else:
                    # below this new branch add the subtree
                    # make a new copy of attrs without A
                    new_attrs = [attr for attr in attrs if attr != A] 
                    child = self.ID3(branch_examples, target_attr, new_attrs, depth + 1)
                    new_branch.children.append(child)

        return root


    def print_tree(self, root: Node, depth):
        '''Print decision tree'''

        print("\t"*(depth+1), end='')
        if root.is_leaf():
            print(" -> ", end='')
        print(root.value)
        for child in root.children:
            self.print_tree(child, depth + 1)


    def find_boundaries(self, data: pd.DataFrame):
        '''Return boundaries that separate data to equal distance intervals'''

        # find max, min values in feature
        max_val = data.max()
        min_val = data.min()

        # calculate width for equal distance intervals
        width = (max_val - min_val) / NUM_BINS

        # get boundaries
        boundaries = [(min_val + (i+1) * width) for i in range(NUM_BINS - 1)]
        
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


    def discretize_data(self, data: pd.DataFrame, features_to_discretized: list):
        '''Return equal distance discretized dataset of given continuous dataset'''

        discretized_data = data.copy()
        # equal distance discretize each feature
        for col in features_to_discretized:
            # typecast column from continuous data to discrete data
            discretized_data[col] = discretized_data[col].astype(int)
            
            for row in range(len(data)): # for each feature value
                # update entry with corresponding interval number
                discretized_data.at[row, col] = self.get_discretize_value(
                            discretized_data.at[row, col], self.boundaries[col])
        
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

        test_features = test_data.columns.tolist()
        discretized_test_data = self.discretize_data(test_data, test_features[:8])
        
        classifications = [] # initialize list of predictions for each datum

        for i in range(len(discretized_test_data)):
            classifications.append(self.classify_datum(
                                self.root, discretized_test_data.iloc[i, :]))
        
        return pd.DataFrame(data=classifications, columns=['Prediction'])


    def training_set_error(self, test_data: pd.DataFrame, test_label_key: pd.DataFrame):
        '''Return number of incorrect predictions made by decision tree'''

        # get prediction data 
        predictions = self.classify_data(test_data) 

        compare = pd.concat([predictions, test_label_key], axis=1, join='inner')
        compare.to_csv(path_or_buf=f'classified_pokemon.csv', index=False)

        incorrect_ct = 0 # count incorrect predictions
        # compare prediction with test key
        for i in range (len(test_label_key)):
            if predictions.iat[i, 0] != test_label_key.iat[i, 0]: incorrect_ct += 1

        return incorrect_ct


def main():
    # read data
    features = pd.read_csv(FEATURE_FILE)   
    labels = pd.read_csv(LABEL_FILE)

    # get training dataset
    data = pd.concat([features, labels], axis=1, join='inner')

    # make decision tree
    tree = DecisionTreeClassifier(data, labels.columns[0], list(features.columns))
    #tree.print_tree(tree.root, 0)

    # calculate predictions accuracy
    error_ct = tree.training_set_error(features, labels)
    data_ct = len(data)
    error_rate = error_ct / data_ct
    print(f'Training set error of tree = {error_ct}/{data_ct} = {error_rate}')    
    print('Training set accuracy of tree =', str(1 - error_rate))   

main()