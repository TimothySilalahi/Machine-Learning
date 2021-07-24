import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#train the data set with Linear Regrresion
from sklearn.linear_model import LinearRegression

LR=LinearRegression()
LR.fit(x,y)

#train the dataset with Polynomial Regression
##transform dataset into polynomial
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=5)
x_poly = poly.fit_transform(x)
##train the new polynomial features to a linear model
LR_2 = LinearRegression()
LR_2.fit(x_poly,y)

#visualise results Linear Regression
plt.scatter(x,y, color= 'red')
plt.plot(x,LR.predict(x), color='blue')
plt.title('Linear Regression Model (Truth or Bluff)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

#visualise results Polynomial Regression
x_grid = np.arange(min(x),max(x),1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y, color= 'red')
plt.plot(x_grid, LR_2.predict(poly.fit_transform(x_grid)), color = 'blue')
plt.title('Polynomial Regression Model (Truth or Bluff)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

#predict using linear model
print(LR.predict([[6.5]]))

#predict using polynomial model
print(LR_2.predict(poly.fit_transform([[6.5]])))