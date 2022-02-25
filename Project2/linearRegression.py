# Project 2: Linear and Polynomial Regression
# Author: Duong Hoang
# CS 460G - 001
# Due Date: March 6th, 2022

'''
    Purpose: predict the quality of red wine based on 11 features describing 
            a particular vintage using linear regression with gradient descent
    Pre-cond: a wine quality data csv file
    Post-cond: 

'''


### Implementation ###

# initialize
DATA_FILE = 'winequality-red'
NUM_FEATURES = 11
ALPHA = 0

# import libraries
from gettext import find
import pandas as pd
import numpy as np

class Regression():
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.feature_names = data.columns.tolist()[:NUM_FEATURES]
        self.normalized_data = self.normalize_data(data)


    def find_limits(self, data: pd.DataFrame):
        '''Return a list of [min, max] pair of each columns'''
        limits = []
        for col in range(len(data) - 1):
            limits.append(min(data[col]), max(data[col]))

        return limits
            
    def normalize_data(self, data: pd.DataFrame):
        '''Normalize feature values to range 0-1'''
        normalized_data = data.copy()
        limits = self.find_limits(data)

        for row in normalized_data:
            for col in range(len(normalized_data) - 1):
                row[col] = (row[col] - limits[col][0]) / (limits[col][1] - limits[col][0])

        return normalized_data

    def predict_data(self, data: pd.DataFrame):
        predictions = []
        #FIXME

        return pd.DataFrame(data=predictions)

    def calculate_MSE(self, predictions: pd.DataFrame, keys: pd.DataFrame):
        num_examples = len(keys)
        sum_errors = 0

        for row in range(num_examples):
            predict_error = predictions[row, 0] - keys[row, 0]
            sum_errors += predict_error ** 2
        
        return np.sqrt(sum_errors / num_examples)


def main():
    # read data
    data = pd.read_csv(f'{DATA_FILE}.csv')   
    print(len(data))

main()