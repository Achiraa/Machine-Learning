# importing libraries 
#Data Pre-processing
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
#from sklearn.linear_model import LinearRegression 

data_set=pd.read_csv('Salary_Data.csv')

#extracting data in X and Y
x= data_set.iloc[:, :-1].values  
y= data_set.iloc[:, 1].values

#splitting the data
from sklearn.model_selection import train_test_split  

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3, random_state=0)
#Fitting the Simple Linear Regression model to the training dataset 
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_train, y_train)  

#Prediction of Test and Training set result  

y_pred= regressor.predict(x_test)  
x_pred= regressor.predict(x_train) 
print(y_pred)
print(x_pred)


df=pd.DataFrame(y_test,y_pred)

#Plotting graphs
plt.scatter(x_train, y_train, color="green")   
plt.plot(x_train, x_pred, color="red")    
plt.title("Salary vs Experience (Training Dataset)")  
plt.xlabel("Years of Experience")  
plt.ylabel("Salary(In Rupees)")  
plt.show() 
