# Data collection
# import pandas_datareader as pdr
# key = 'c9dd08afd9a2d765ee481c585acccce430dae8e2'
# df = pdr.get_data_tiingo('AAPL', api_key=key)
# df.to_csv('APPL.csv')

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
import pandas as pd
df = pd.read_csv(dir_path + "/StockPricePrediction-DL/APPL.csv")

df1 = df.reset_index()['close']

import matplotlib.pyplot as plt
plt.plot(df1)

# LSTM sensitive to scale of data, so we apply minmax scalar
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

# Split data into train and test
training_size = int(len(df1)*0.65)
test_size = len(df1) - training_size
train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]


# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    datax, datay = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        datax.append(a)
        datay.append((dataset[i+time_step, 0]))
    return np.array(datax), np.array(datay)


# Reshape data into X=t,t+1,t+2,t+3 and y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape the inputs to be [samples, time steps, features] which is required fro LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
train_mse = math.sqrt(mean_squared_error(y_train, train_predict))
test_mse = math.sqrt(mean_squared_error(y_test, test_predict))

# Plotting
# Shift train prediction for plotting
look_back = 100
trainpredictplot = np.empty_like(df1)
trainpredictplot[:, :] = np.nan
trainpredictplot[look_back:len(train_predict)+look_back, :1] = train_predict
# Shift test prediction for plotting
testpredictplot = np.empty_like(df1)
testpredictplot[:, :] = np.nan
testpredictplot[len(train_predict)+(look_back*2)+1:len(df1)-1,:] = test_predict
# Plot baseline and predictors
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainpredictplot)
plt.plot(testpredictplot)
plt.show()

# Predicting for 30 days of data
x_input = test_data[len(test_data)-100:].reshape(1, -1)
print(x_input.shape)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

print(len(temp_input))

# Demostrate prediction for next 30 days
lst_output = []
n_steps = 100
i = 0
while i < 30:
    if len(temp_input) > 100:
        x_input = np.array(temp_input[1:])
        print(' {} day input {}'.format(i, x_input))
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape(1, n_steps, -1)
        yhat = model.predict(x_input, verbose=0)
        print(' {} day output {}'.format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
    i = i + 1
print(lst_output)

day_new = np.arange(1, 101)
day_pred = np.arange(101, 131)
plt.plot(day_new, scaler.inverse_transform(df1[1159:]).reshape(-1))
plt.plot(day_pred, scaler.inverse_transform(lst_output).reshape(-1))

df3 = df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])

df3 = scaler.inverse_transform(df3).tolist()
plt.plot(df3)