#Data Pre-procesing Step  
# importing libraries  
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
  
#importing datasets  
data_set= pd.read_csv('framingham.csv')
print(data_set.isnull().sum())

#Replace the missing values for numerical columns with mean
data_set['education'] = data_set['education'].fillna(data_set['education'].mean())
data_set['cigsPerDay'] = data_set['cigsPerDay'].fillna(data_set['cigsPerDay'].mean())
data_set['BPMeds'] = data_set['BPMeds'].fillna(data_set['BPMeds'].mean())
data_set['totChol'] = data_set['totChol'].fillna(data_set['totChol'].mean())
data_set['BMI'] = data_set['BMI'].fillna(data_set['BMI'].mean())
data_set['heartRate'] = data_set['heartRate'].fillna(data_set['heartRate'].mean())
data_set['glucose'] = data_set['glucose'].fillna(data_set['glucose'].mean())


#Extracting Independent and dependent Variable  
x= data_set.iloc[:, 5:10].values  
y= data_set.iloc[:, -1].values 
#print(x)
#print(y)

# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0) 
print(x_train,y_train)
print(x_test,y_test)

#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test) 

#Fitting Logistic Regression to the training set  
from sklearn.linear_model import LogisticRegression  
classifier= LogisticRegression(random_state=0)  
classifier.fit(x_train, y_train) 

#Predicting the test set result  
y_pred= classifier.predict(x_test)  

#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test,y_pred) 
print ("Confusion Matrix : \n", cm)

from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(y_test, y_pred))
