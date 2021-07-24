# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#create dendogram
from scipy.cluster import hierarchy as sch

dendogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendogram')
plt.xlabel('customers')
plt.ylabel('Distance')
plt.show()

#build train and run hierarchical clustering model
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1], s=100, c='red',label='Cluster 0')
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1], s=100, c='blue',label='Cluster 1')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1], s=100, c='green',label='Cluster 2')
plt.title('KMeans Clustering using Elbow Method')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()


