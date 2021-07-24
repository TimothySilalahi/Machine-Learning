#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#split test set lalala
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)


#fit the regression model
from sklearn.linear_model import LinearRegression

LR = LinearRegression()
LR.fit(x_train,y_train)

#predicting
y_pred = LR.predict(x_test)

#plotting
#plotting the dots (scatter the dots)
plt.scatter(x_train,y_train, color='red')
#plotting the line
plt.plot(x_train, LR.predict(x_train), color='blue')
plt.title('Salary Vs Experience (training)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#plotting test

plt.scatter(x_test,y_test,color='red')

plt.plot(x_test,y_pred, color='blue')
plt.title('Salary Vs Experience (test)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

pred12 = LR.predict([[20]])
print(pred12)

print(LR.coef_)
print(LR.intercept_)


