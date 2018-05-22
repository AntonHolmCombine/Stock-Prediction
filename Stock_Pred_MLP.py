import xlrd
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Conv1D, MaxPooling1D, Flatten
from sklearn import preprocessing
import matplotlib.pyplot as plt

def gen_sequence(data,seq_length):
	num_elements = data.shape[0]
	for start, stop in zip(range(0,num_elements-seq_length), range(seq_length,num_elements)):
		yield data[start:stop]

data = pd.read_excel("DataSheet_AZ.xlsx")

prices = data["Slutkurs"].values

window_length = 20

data_seq = list(gen_sequence(prices, window_length))

data_train = np.zeros(shape=(len(data_seq),window_length))
y_data = np.zeros(shape=(data_train.shape[0],1))

for i in range(0,(data.shape[0]-window_length)):
	data_train[i, ] = data_seq[i]
	y_data[i] = prices[i+window_length]

data_train = np.expand_dims(data_train,axis=2)
y_max = np.max(y_data)
y_min = np.min(y_data)

tp = 2*(y_data-y_min)
tp = tp/(y_max-y_min)
tp = tp - 1
y_data = tp

x_mean = np.mean(prices)
x_std = np.std(prices)

tp = data_train - x_mean
tp = tp/x_std
data_train = tp

doStuff = True

if(doStuff):
	


	model = Sequential()

	model.add(Conv1D(input_shape=(window_length,1),filters = 32,kernel_size=3, activation="tanh",padding="same"))
	model.add(Dropout(0.2))
	model.add(MaxPooling1D())
	model.add(Conv1D(filters=16,kernel_size=2,activation="tanh",padding="same"))
	model.add(Flatten())
	model.add(Dense(units=1,activation="tanh"))
	model.compile(loss="mean_squared_error",optimizer="adam",metrics=["mae"])

	#model.add(LSTM(input_shape=(window_length,1),units=200,return_sequences=True))
	#model.add(Dropout(0.2))
	#model.add(LSTM(units=100,return_sequences=True))
	#model.add(Dropout(0.2))
	#model.add(LSTM(units=100,return_sequences=False))
	#model.add(Dense(units=1,activation="tanh"))
	#model.compile(loss="mean_squared_error",optimizer="adam",metrics=["mae"])

	model.fit(data_train,y_data,batch_size=10,epochs=300,verbose=1,shuffle=True)

	predictions = model.predict(data_train)


	plt.figure(figsize=(15,5))
	plt.plot(y_data,color="red",label="True values")
	plt.plot(predictions,color="blue",label="Predictions")
	plt.xlabel("Time")
	plt.ylabel("Normalized Price")
	plt.legend(loc="best")
	plt.show()
