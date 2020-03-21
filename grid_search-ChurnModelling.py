# ANN

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
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

#building a classifier function
def build_class(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = "uniform",activation="relu",input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = "uniform",activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = "uniform", activation= "sigmoid"))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy',metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_class, batch_size = 10, epochs = 100)
parameters = {'batch_size': [25,32],'epochs':[100,500],'optimizer':['adam','rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_accuracy_