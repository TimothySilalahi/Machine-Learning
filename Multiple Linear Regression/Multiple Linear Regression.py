import pandas as pd
import numpy as np


#import data

dataset = pd.read_csv('50_Startups.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#encoding categorical values
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x= np.array(ct.fit_transform(x))

#splitting dataset to training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

#create a regression
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_train,y_train)

#predict
y_pred = LR.predict(x_test)
#set precision (jadi 2 dibelakang koma)
np.set_printoptions(precision=2)
#compare y_pred (predicted results with model) with y_test (actual result)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),axis=1))

#predicting
print(LR.predict([[1,0,0,160000,130000,300000]]))
