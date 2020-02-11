# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:44:43 2019

@author: Muhammad Hassan Ur Rehman
"""

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
#encoding categorical data
from sklearn.preprocessing import LabelEncoder
#lABEL Encoder is used because it is easy to convert for machine to read number other then string so 
#we converted country names into numbers

labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
# but now the problem is that these numbers while training would compare themselve and 
#due to which they will give precedence 
#so we will use boolean
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
#splitting the dataset into training set and testing seen

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=0)
# if machine learn anything by heart and couldn't understand properly means he is overfitting
