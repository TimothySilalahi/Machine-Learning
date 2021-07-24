import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#y = y.reshape(len(y),1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#create clasifier
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(x_train,y_train)

#predict single record
print(LR.predict(sc.transform([[30,87000]])))

#predict test set (record(s))
y_pred = LR.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#make confusion matrix
from sklearn.metrics import confusion_matrix


matrix = confusion_matrix(y_test,y_pred)


from sklearn.metrics import accuracy_score

score = accuracy_score(y_test,y_pred)
print(score)

