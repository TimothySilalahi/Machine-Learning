import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing data set
dataset = pd.read_csv('breast_cancer.csv')

x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#split dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)
print(x_test)

#train classifier
from sklearn.linear_model import LogisticRegression
cf = LogisticRegression(random_state=0)

cf.fit(x_train,y_train)
#predict
y_pred = cf.predict(x_test)

#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


#scoring
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))