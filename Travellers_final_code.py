# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:43:21 2018

@author: kabil
"""

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier


# importing packages for visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

#importing lightgbm package
import lightgbm as lgb

#importing data
dataset=pd.read_csv('uconn_comp_2018_train_1024.csv')
dataset.describe()
dataset.info()

#dropping the fields identified as not significant
dataset_train=dataset.drop(['claim_number','zip_code','claim_date','claim_day_of_week','vehicle_color'],axis=1)
dataset_train.info()

#convert data frame into object
x=dataset_train.iloc[:,:18].values

#convert into dataframe
df_x = pd.DataFrame(x)
y=dataset_train.iloc[:,19].values
print(y)


#encoding
from sklearn.preprocessing import LabelEncoder
label_x = LabelEncoder()
x[:,1] =label_x.fit_transform(x[:,1])
x[:,7] =label_x.fit_transform(x[:,7])
x[:,8] =label_x.fit_transform(x[:,8])
x[:,12] =label_x.fit_transform(x[:,12])
x[:,16] =label_x.fit_transform(x[:,16])

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[1,5,6,7,8,12,13,16],sparse=False,dtype=np.int64)
x=onehotencoder.fit_transform(x).toarray()


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#converting xtrain to a dataframe
df_xtrain = pd.DataFrame(x_train)

#Prediction
d_train = lgb.Dataset(x_train, label=y_train)
#paramters
params = {}
params['learning_rate'] = 0.1
params['boosting_type'] = 'dart'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 31
params['min_data'] = 50
params['max_depth'] = 10

clf = lgb.train(params, d_train, 100)

#predictions for test data
y_pred=clf.predict(x_test)

#convert into binary values
for i in range(0,5999):
    if y_pred[i]>=.5:       # setting threshold to .5
       y_pred[i]=1
    else:  
       y_pred[i]=0


#Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,y_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)












