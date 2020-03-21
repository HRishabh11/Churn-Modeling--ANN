# Best ANN Classifier

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
data = pd.read_csv("Churn_Modelling.csv")
X = data.iloc[:,3:13]
y = data.iloc[:,-1]

#Encoding Categorical data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
column_transformer = ColumnTransformer([('encoder', OneHotEncoder(categories = [['Spain','France','Germany'],['Male','Female']]),[1,2])], remainder='passthrough')
X = column_transformer.fit_transform(X)
X = X[:,1:]
X = np.delete(X,2,axis = 1)

#splitting the data into train-test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0 )

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#importing the keras library
import keras
from keras.models import Sequential
from keras.layers import Dense


#Creating the Classifier 
classifier = Sequential()
classifier.add(Dense(output_dim = 6, init = "uniform",activation="relu",input_dim = 11))
classifier.add(Dense(output_dim = 6, init = "uniform",activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = "uniform", activation= "sigmoid"))
classifier.compile(optimizer = "rmsprop", loss = 'binary_crossentropy',metrics = ['accuracy'])

#Fitting the classifier
classifier.fit(X_train,y_train,batch_size = 32, epochs = 500 )

#predicting 
best_pred = classifier.predict(X_test)
best_pred = (best_pred > 0.5)

#Evaluating
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, best_pred)
cm

#Calculating the accuracy
accuracy = (cm[0,0]+cm[1,1])/cm.sum()
print("The Accuracy of our prediction is {}".format(accuracy))