import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y),1)
#feature scaling
from sklearn.preprocessing import StandardScaler

x_sc = StandardScaler()
y_sc = StandardScaler()

x = x_sc.fit_transform(x)
y = y_sc.fit_transform(y)

#do the SVR
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x,y)

#do prediction
pred = y_sc.inverse_transform(regressor.predict(x_sc.transform([[6.5]])))

#visualize SVM results
plt.scatter(x_sc.inverse_transform(x),y_sc.inverse_transform(y), color= 'red')
plt.plot(x_sc.inverse_transform(x),y_sc.inverse_transform(regressor.predict(x)), color='blue')
plt.title('Support Vector Regression Model (Truth or Bluff)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
#plt.show()

#smoother
x_grid = np.arange(min(x_sc.inverse_transform(x)),max(x_sc.inverse_transform(x)),1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x_sc.inverse_transform(x),y_sc.inverse_transform(y), color= 'red')
plt.plot(x_grid,y_sc.inverse_transform(regressor.predict(x_sc.transform(x_grid))), color='blue')
plt.title('Support Vector Regression Model (Truth or Bluff)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()