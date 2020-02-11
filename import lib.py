# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 22:12:04 2019

@author: hp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values 
Y=dataset.iloc[:,3].values
#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy ='mean',axis=0) 
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3]=imputer.transform(X[:, 1:3])
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:, 0])