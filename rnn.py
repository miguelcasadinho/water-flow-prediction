# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
# Read the csv with pandas
dataset_train = pd.read_csv(('train_set.csv'), delimiter=';')
# Leave only the flow column
training_set = dataset_train.iloc[:, 3:4].values

print(training_set)
