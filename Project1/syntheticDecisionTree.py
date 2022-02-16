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
NUM_BINS = 5 # always greater than or equal to 1
DATA_FILE = 'synthetic-3'
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
  

class DecisionTree(Node):
    def __init__(self, data: pd.DataFrame):
        self.training_data = data
        self.label_index = len(data.columns) - 1
        self.features_list = (data.columns.tolist())[:self.label_index]
        self.boundaries = [self.find_boundaries(data.iloc[:, feature]) for feature in self.features_list]
        self.discretized_data = self.discretize_data(data, self.features_list)
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
            for value in range(NUM_BINS):
                # add a new tree branch below root, corresponding to the test A = vi
                new_branch = Node(value)
                root.children.append(new_branch)
                # let examples(vi) be the subset of examples that have vi for A
                branch_examples = examples.loc[examples.iloc[:, A] == value]
                # if examples(vi) is empty or tree about to reach max depth,
                if len(branch_examples) == 0 or depth == MAX_DEPTH - 1:
                    # then below this branch add a leaf node with 
                    # label = most common val of targetAttr in examples
                    leaf = Node(examples.iloc[:, target_attr].value_counts().idxmax())
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
            discretized_data.iloc[:, col] = discretized_data.iloc[:, col].astype(int)

            for row in range(len(data)): # for each feature value
                # update entry with corresponding interval number
                discretized_data.at[row, col] = self.get_discretize_value(
                                    data.at[row, col], self.boundaries[col])
        
        return discretized_data


    def classify_datum(self, current: Node(), datum: list):
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
        
        # discretize test data
        test_data_features = test_data.columns.tolist()
        discretized_test_data = self.discretize_data(test_data, test_data_features).values.tolist()
        
        classifications = [] # initialize list of predictions for each datum

        # classify each datum using decision tree
        for i in range(len(discretized_test_data)):
            classifications.append(self.classify_datum(self.root, discretized_test_data[i]))
        
        return pd.DataFrame(data=classifications)


    def calculate_accuracy(self, test_data: pd.DataFrame, test_label_key: pd.DataFrame):
        '''Return accuracy rate of decision tree predictions'''

        # compare datasets as lists of labels
        predictions = self.classify_data(test_data) 

        # create csv file with first column is prediction, second column is key
        compare = pd.concat([predictions, test_label_key], axis=1, join='inner')
        compare.to_csv(path_or_buf=f'classified_{DATA_FILE}.csv', index=False)

        correct_ct = 0 # count correct predictions
        data_ct = len(test_label_key) # total number of data to compare
        # compare prediction with test key
        for i in range (data_ct):         
            if predictions.iat[i, 0] == test_label_key.iat[i, 0]: correct_ct += 1

        return correct_ct / data_ct


    def get_tree_leaves(self, current: Node, feature_1, feature_2, classifications: list):
        # get all classifications in the form:
        # {feature_1: interval_no, feature_2: interval_no, class_label: label}
        
        
        if current.is_leaf():
            classifications.append([feature_1, feature_2, current.value])
        else: 
            for child in current.children:
                if current.value == 0:
                    feature_1 = child.value
                else:
                    feature_2 = child.value
                self.get_tree_leaves(child.children[0], feature_1, feature_2, classifications)

        return classifications


    def visualize_data(self):
        '''Scatterplot visualization of training data and 
        decision tree's classifications color coded in background
        '''

        self.training_data.columns = ['feature_1', 'feature_2', 'class_label']
        colors = {0: 'red', 1: 'blue'} # class label color dictionary

        # create training data scatter plot color coded by class label
        axes = plt.gca()
        groups = self.training_data.groupby('class_label')
        for key, group in groups:
            group.plot(ax=axes, kind='scatter', x='feature_1', y='feature_2', 
            label=key, color=colors[key])
        plt.legend()
        plt.margins(x=0, y=0)

        # get x gridlines based on feature 1 boundaries and x-axis limits
        x_left, x_right = axes.get_xlim()
        #x_left -= MARGIN
        #x_right += MARGIN
        x_grids = (self.boundaries[0])[:] # get a copy of feature 1 boundaries
        x_grids.insert(0, x_left)
        x_grids.append(x_right)
        print(x_grids)

        # get y gridlines based on feature 2 boundaries and y-axis limits
        y_left, y_right = axes.get_ylim()
        #y_left -= MARGIN*2
        #y_right += MARGIN*2
        y_grids = (self.boundaries[1])[:] # get a copy of feature 2 boundaries
        y_grids.insert(0, y_left)
        y_grids.append(y_right)
        print(y_grids)

        # get all classifications
        classifications = []
        classifications = self.get_tree_leaves(self.root, INVALID_VALUE, 
                                                INVALID_VALUE, classifications)

        # fill areas limited by gridlines based on decision tree classifications
        for classification in classifications:
            # initialize grid values
            Xs = []
            Ys = []
            # get x-axis gridlines
            if classification[0] == INVALID_VALUE: Xs.extend((x_left, x_right))
            else:
                Xs.extend((x_grids[classification[0]], x_grids[classification[0] + 1]))
            # get y-axis gridlines
            if classification[1] == INVALID_VALUE: Ys.extend((y_left, y_right))
            else:
                Ys.extend((y_grids[classification[1]], y_grids[classification[1] + 1]))
            # color area restricted by x and y gridlines
            plt.fill_between(x=Xs, y1=Ys[0], y2=Ys[1], 
                            facecolor=colors[classification[2]], alpha =0.3)

        # display plot and export to file
        plt.show()

        # save plot
        plt.savefig(f'visualize_{DATA_FILE}.png')


def main():
    # read data
    data = pd.read_csv(f'{DATA_FILE}.csv', header=None)   
    label_index = len(data.columns) - 1
    test_data = pd.DataFrame(data.iloc[:, :label_index])
    test_label_key = pd.DataFrame(data.iloc[:, label_index])

    # make decision tree
    tree = DecisionTree(data)
    tree.printTree(tree.root, 0)

    # calculate predictions accuracy
    print('Accuracy of tree =', tree.calculate_accuracy(test_data, test_label_key))    

    # plot 
    tree.visualize_data()
main()