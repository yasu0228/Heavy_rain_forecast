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

X_test = test[:, 0:7]
Y_test = test[:, 8]

print('X_test:',X_test)
print('Y_test:',Y_test)

###seikika###
# sc = preprocessing.StandardScaler()
sc = sc = preprocessing.MinMaxScaler()
sc.fit(X)
X = sc.transform(X)
sc.fit(X)
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
history = model.fit(X,Y,batch_size=5, epochs=100, validation_data=(X_test,Y_test),verbose=2)
                 
X_eval = X_test[0:50]

Y_eval = Y_test[0:50]

print('X_eval:',X_eval)
print('Y_eval:',Y_eval)
                 
t_pred = model.predict(X_eval)
t_pred = np.reshape(t_pred, (1, len(t_pred)))
pred = model.predict(X_test,batch_size = 5)

print('t_pred:',t_pred)
print('pred:',pred)
                 
#mae = abs(pred - Y_test)
#pct = (abs(pred - Y_test) / Y_est) * 100
#compare = np.stack([Y_test, pred, mae, pct], -1)
                 
#np.savetxt(r'C:\Users\koizum\Documents\IBSM\data\out.csv', compare, delimiter=',',header='Y_test,Y_pred,mae,pct')
####
                 
                 
model_file=r'\model_architecture_timeaverage.json'
weights_file=r'\model_weights_timeaverage.h5'
# VM内CPU利用率予測用モデル
model_file_guestcpu=r'model_architecture_guestcpuuseave.json'

weights_file_guestcpu=r'model_weights_guestcpuuseave.h5'

def save_model(model,model_file,weights_file):
    # saving model
    json_model = model.to_json()
    open(model_file, 'w').write(json_model)
    # saving weights
    model.save_weights(weights_file,overwrite=True)
                 
save_model(model,model_file_guestcpu,weights_file_guestcpu)
###check load model###
                 
####
                 
# %matplotlib inline
# yyplot 作成関数
yvalues = np.concatenate([Y_test.flatten(),pred.flatten()])
ymin, ymax, yrange = np.amin(yvalues), np.amax(yvalues),np.ptp(yvalues)
fig = plt.figure(figsize=(8,8))
plt.scatter(Y_test, pred, c='gray')
plt.plot([ymin - yrange * 0.01, ymax + yrange * 0.01], [ymin - yrange * 0.01, ymax + yrange * 0.01], c='gray')


if ((ymin - yrange * 0.01) is None) or ((ymax + yrange * 0.01) is None):
#if ((ymin - yrange * 0.01) is Nan) or ((ymax + yrange * 0.01) is Nan):
	plt.xlim(0,0)
else:
	plt.xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
	
#if ((ymin - yrange * 0.01) is None) or ((ymax + yrange * 0.01) is None): 
#	plt.ylim(0,0)
#else:
plt.ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)



	
plt.xlabel('Precipitation amount',fontsize=24)
plt.ylabel('Precipitation amount',fontsize=24)
plt.title('Precipitation amount',fontsize=24)

plt.tick_params(labelsize=16)
plt.show()


val, idx = min((val,idx) for (idx, val) in enumerate(history.history['val_loss']))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model,loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

     
        
