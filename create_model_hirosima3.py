import keras.backend as K
import pandas 
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing #uc
import pdb
import matplotlib.pyplot as plt
import csv
from keras import regularizers
from keras.models import load_model
import codecs


####evaluate-function########
datafram = pandas.read_csv(r"hiroshima20180706-2.csv")
with codecs.open("D-hiroshima20180705-7.csv","r","Shift-JIS","ignore") as file:
	dataframe = pandas.read_csv(file,delimiter=",")

dataset = dataframe.values
print ('dataset size is {0}'.format(dataset.shape))
train,test = train_test_split(dataset,test_size = 0.2)

X = train[:, 0:7]

Y =train[:, 8]
print ('Y size is {0}'.format(Y.shape))
print ('X size is {0}'.format(X.shape))
print(Y[0])

X_test = [[1,1,1,1,1,1,1]]
#X_test = test[:, 0:7]
#Y_test = test[:, 8]

#print('X_test:',X_test)
#print('Y_test:',Y_test)

###seikika###
# sc = preprocessing.StandardScaler()
sc = sc = preprocessing.MinMaxScaler()
sc.fit(X)
X = sc.transform(X)
sc.fit(X_test)
X_test = sc.transform(X_test)
                 
             
# define base model
model = Sequential()
model.add(Dense(16, input_dim=7, kernel_initializer='normal',activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(output_dim=1, kernel_initializer='normal'))
                 # Compile model
model.compile(loss='mape', optimizer='adam',metrics=['mae','mape']) # 'mean_squared_error','mean_absolute_percentage_error' mean_squared_logarithmic_error
#history = model.fit(X,Y,batch_size=5, epochs=100, validation_data=(X_test,Y_test),verbose=2)

history = model.fit(X,Y,batch_size=20, epochs=100,verbose=2)
X_eval = X_test[0:1]
                 
#X_eval = X_test[0:50]

#Y_eval = Y_test[0:50]

#print('X_eval:',X_eval)
#print('Y_eval:',Y_eval)
                 
t_pred = model.predict(X_eval)
t_pred = np.reshape(t_pred, (1, len(t_pred)))
#pred = model.predict(X_test,batch_size = 5)

print('t_pred:',t_pred)

  