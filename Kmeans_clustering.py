#importing libraries 
#Data Pre-processing
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
#from sklearn.linear_model import LinearRegression 

data_set=pd.read_csv('Mall_Customers.csv')

#extracting data in X and Y
x= data_set.iloc[:, [2,3]].values
#print(x)

#Cluster is the module, KMeans is package
from sklearn.cluster import KMeans

wcss_list=[] #initializing the list for the value of WCSS

#Using for loop fon iterations from 1 to 10
for i in range(1, 11):  

    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)  

    kmeans.fit(x)  

    wcss_list.append(kmeans.inertia_)  

plt.plot(range(1, 11), wcss_list)
plt.title('The Elobw Method Graph')  
plt.xlabel('Number of clusters(k)')  
plt.ylabel('wcss_list')  
plt.show()

#training the K-means model on a dataset  

kmeans = KMeans(n_clusters=4, init='k-means++', random_state= 42)  

y_predict= kmeans.fit_predict(x)
print(y_predict)

#visualizing the clusters  

plt.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s = 100, c = 'blue', label = 'Cluster 1') #for first cluster  
plt.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s = 100, c = 'green', label = 'Cluster 2') #for second cluster
plt.scatter(x[y_predict== 2, 0], x[y_predict == 2, 1], s = 100, c = 'red', label = 'Cluster 3') #for third cluster  
plt.scatter(x[y_predict == 3, 0], x[y_predict == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4') #for fourth cluster  
#plt.scatter(x[y_predict == 4, 0], x[y_predict == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5') #for fifth cluster  
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')   
plt.title('Clusters of customers')  
plt.xlabel('Annual Income (k$)')  
plt.ylabel('Spending Score (1-100)')  
plt.legend()  
plt.show()