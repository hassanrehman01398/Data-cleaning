# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
# WE CAN USE
#MOST FREQUENT
#MEAN
#MEDIAN
imputer = imputer.fit(X[:, 1:3])
#transform will replace the missing data 
X[:, 1:3] = imputer.transform(X[:, 1:3])