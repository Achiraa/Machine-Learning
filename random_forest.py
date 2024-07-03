import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

data_set=pd.read_csv('User_Data.csv')

#extracting data in X and Y
x= data_set.iloc[:, [2,3]].values  
y= data_set.iloc[:, 4].values 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= .25, random_state=0)
print(x_train)
#feature scaling , to acheive the graph we need this feature scaling
#converting the actucal vblaue to -1  and +1
from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)
#we won't use .fit_tranfrom for test model because we aren't using it for prediction,  just printing it as it is.
x_test=st_x.transform(x_test)


#print(x_test)
#print(x_train)
#Fitting RandomForest Classifier to the Training Set
from sklearn.ensemble import RandomForestClassifier  
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
classifier.fit(x_train, y_train)

y_pred=classifier.predict(x_test)
#y_pred_train=classifier.predict(x_train)


#We always predict last col in data set
#Use only continous data not categorical data

#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test,y_pred) 
print ("Confusion Matrix : \n", cm)

#for regression only not for classification
#r**2 is used for accuracy in regression

#Performance measure–Accuracy
#Supervised Learning
from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(y_test, y_pred))

#accuracy is measured as percent , 0.87 is 87%
#here 38 out of 79 is predicted data and 6 and 11 is wrong prediction in confusion matrix
