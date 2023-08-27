#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('mode.chained_assignment', None)

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import RMSprop
from keras.callbacks import Callback


# In[30]:


humidity = pd.read_csv("/Users/shyleshraju/Downloads/archive (5)/humidity.csv")
temp = pd.read_csv("/Users/shyleshraju/Downloads/archive (5)/temperature.csv")
pressure = pd.read_csv("/Users/shyleshraju/Downloads/archive (5)/pressure.csv")


# In[31]:


humidity_SF = humidity[['datetime','San Francisco']]
temp_SF = temp[['datetime','San Francisco']]
pressure_SF = pressure[['datetime','San Francisco']]
humidity_SF.head(10)


# In[32]:


humidity_SF.tail(10)


# In[33]:


print(humidity_SF.shape)
print(temp_SF.shape)
print(pressure_SF.shape)


# In[34]:


print("NaN in the humidity dataset",humidity_SF.isna().sum()['San Francisco'])
print("NaN in the temperature dataset",temp_SF.isna().sum()['San Francisco'])
print("NaN in the pressure dataset",pressure_SF.isna().sum()['San Francisco'])


# In[35]:


Tp = 36203


# In[36]:


def plot_train_points(quantity='humidity',Tp=36203):
    plt.figure(figsize=(15,4))
    if quantity=='humidity':
        plt.title("Humidity of first {} data points".format(Tp),fontsize=16)
        plt.plot(humidity_SF['San Francisco'][:Tp],c='k',lw=1)
    if quantity=='temperature':
        plt.title("Temperature of first {} data points".format(Tp),fontsize=16)
        plt.plot(temp_SF['San Francisco'][:Tp],c='k',lw=1)
    if quantity=='pressure':
        plt.title("Pressure of first {} data points".format(Tp),fontsize=16)
        plt.plot(pressure_SF['San Francisco'][:Tp],c='k',lw=1)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
plot_train_points('humidity')


# In[37]:


plot_train_points('temperature')


# In[38]:


plot_train_points('pressure')


# In[39]:


humidity_SF.interpolate(inplace=True)
humidity_SF.dropna(inplace=True)

temp_SF.interpolate(inplace=True)
temp_SF.dropna(inplace=True)

pressure_SF.interpolate(inplace=True)
pressure_SF.dropna(inplace=True)
print(humidity_SF.shape)
print(temp_SF.shape)
print(pressure_SF.shape)


# In[40]:


train = np.array(humidity_SF['San Francisco'][:Tp])
test = np.array(humidity_SF['San Francisco'][Tp:])
print("Train data length:", train.shape)
print("Test data length:", test.shape)

train=train.reshape(-1,1)
test=test.reshape(-1,1)
plt.figure(figsize=(15,4))
plt.title("Train and test data plotted together",fontsize=16)
plt.plot(np.arange(Tp),train,c='blue')
plt.plot(np.arange(Tp,45252),test,c='orange',alpha=0.7)
plt.legend(['Train','Test'])
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# In[41]:


step = 8
# add step elements into train and test
test = np.append(test,np.repeat(test[-1,],step))
train = np.append(train,np.repeat(train[-1,],step))
print("Train data length:", train.shape)
print("Test data length:", test.shape)


# In[42]:


def convertToMatrix(data, step):
    X, Y =[], []
    for i in range(len(data)-step):
        d=i+step  
        X.append(data[i:d,])
        Y.append(data[d,])
    return np.array(X), np.array(Y)
trainX,trainY =convertToMatrix(train,step)
testX,testY =convertToMatrix(test,step)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print("Training data shape:", trainX.shape,', ',trainY.shape)
print("Test data shape:", testX.shape,', ',testY.shape)


# In[43]:


def build_simple_rnn(num_units=128, embedding=4,num_dense=32,lr=0.001):
    """
    Builds and compiles a simple RNN model
    Arguments:
              num_units: Number of units of a the simple RNN layer
              embedding: Embedding length
              num_dense: Number of neurons in the dense layer followed by the RNN layer
              lr: Learning rate (uses RMSprop optimizer)
    Returns:
              A compiled Keras model.
    """
    model = Sequential()
    model.add(SimpleRNN(units=num_units, input_shape=(1,embedding), activation="relu"))
    model.add(Dense(num_dense, activation="relu"))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=lr),metrics=['mse'])
    
    return model
model_humidity = build_simple_rnn(num_units=128,num_dense=32,embedding=8,lr=0.0005)


# In[44]:


model_humidity.summary()


# In[45]:


class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % 50 == 0 and epoch>0:
            print("Epoch number {} done".format(epoch+1))

batch_size=8
num_epochs = 1000

model_humidity.fit(trainX,trainY, 
          epochs=num_epochs, 
          batch_size=batch_size, 
          callbacks=[MyCallback()],verbose=0)


# In[46]:


plt.figure(figsize=(7,5))
plt.title("RMSE loss over epochs",fontsize=16)
plt.plot(np.sqrt(model_humidity.history.history['loss']),c='k',lw=2)
plt.grid(True)
plt.xlabel("Epochs",fontsize=14)
plt.ylabel("Root-mean-squared error",fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# In[47]:


# Extract RMSE losses and find the minimum loss
rmse_losses = np.sqrt(model_humidity.history.history['loss'])
min_rmse_loss = np.min(rmse_losses)
min_rmse_epoch = np.argmin(rmse_losses) + 1  # Adding 1 to account for 0-based indexing

# Print the summary
print("Humidity Prediction Model Training Summary:")
print("Minimum RMSE Loss: {:.4f}".format(min_rmse_loss))
print("Epoch at Minimum RMSE Loss: {}".format(min_rmse_epoch))


# In[48]:


plt.figure(figsize=(15,4))
plt.title("Training data",fontsize=18)
plt.plot(trainX[:,0][:,0],c='blue')
plt.grid(True)
plt.show()


# In[49]:


trainPredict = model_humidity.predict(trainX)
testPredict= model_humidity.predict(testX)
predicted=np.concatenate((trainPredict,testPredict),axis=0)


# In[59]:


plt.figure(figsize=(10,4))
plt.title("This is what the model predicted or testing data",fontsize=18)
plt.plot(testPredict,c='orange')
plt.grid(True)
plt.show()


# In[52]:


index = humidity_SF.index.values

plt.figure(figsize=(15,5))
plt.title("Humidity: Ground truth and prediction together for humidity",fontsize=18)
plt.plot(index,humidity_SF['San Francisco'],c='blue')
plt.plot(index,predicted,c='orange',alpha=0.75)
plt.legend(['True data','Predicted'],fontsize=15)
plt.axvline(x=Tp, c="r")
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(-20,120)
plt.show()


# In[61]:


index = humidity_SF.index.values

plt.figure(figsize=(15,5))

# Plot Ground Truth Humidity
plt.subplot(2, 1, 1)
plt.title("Ground Truth Humidity", fontsize=18)
plt.plot(index, humidity_SF['San Francisco'], c='blue')
plt.legend(['True data'], fontsize=15)
plt.axvline(x=Tp, c="r")
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(-20, 120)

# Plot Predicted Humidity
plt.subplot(2, 1, 2)
plt.title("Predicted Humidity", fontsize=18)
plt.plot(index[Tp:], predicted[Tp:], c='orange', alpha=0.75)
plt.legend(['Predicted'], fontsize=15)
plt.axvline(x=Tp, c="r")
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(-20, 120)

plt.tight_layout()
plt.show()


# In[69]:


# Calculate mean and standard deviation for denormalization
mean_humidity = humidity_SF['San Francisco'].mean()
std_humidity = humidity_SF['San Francisco'].std()

# Denormalize the predictions and actual values
trainPredict_humidity_denormalized = (trainPredict * std_humidity) + mean_humidity
testPredict_humidity_denormalized = (testPredict * std_humidity) + mean_humidity
trainY_humidity_denormalized = (trainY* std_humidity) + mean_humidity
testY_humidity_denormalized = (testY * std_humidity) + mean_humidity

# Calculate MAPE for humidity predictions
train_mape_humidity = calculate_mape(trainY_humidity_denormalized, trainPredict_humidity_denormalized)
test_mape_humidity = calculate_mape(testY_humidity_denormalized, testPredict_humidity_denormalized)

print("Humidity Model:")
print("Train MAPE: {:.2f}%".format(train_mape_humidity))
print("Test MAPE: {:.2f}%".format(test_mape_humidity))


# In[ ]:


index = humidity_SF.index.values

plt.figure(figsize=(15,5))

# Plot Ground Truth Humidity
plt.subplot(2, 1, 1)
plt.title("Ground Truth Humidity", fontsize=18)
plt.plot(index, humidity_SF['San Francisco'], c='blue')
plt.legend(['True data'], fontsize=15)
plt.axvline(x=Tp, c="r")
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(-20, 120)

# Plot Predicted Humidity
plt.subplot(2, 1, 2)
plt.title("Predicted Humidity", fontsize=18)
plt.plot(index, predicted, c='orange', alpha=0.75)
plt.legend(['Predicted'], fontsize=15)
plt.axvline(x=Tp, c="r")
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(-20, 120)

plt.tight_layout()
plt.show()


# In[53]:


train = np.array(temp_SF['San Francisco'][:Tp])
test = np.array(temp_SF['San Francisco'][Tp:])

train=train.reshape(-1,1)
test=test.reshape(-1,1)

step = 8

# add step elements into train and test
test = np.append(test,np.repeat(test[-1,],step))
train = np.append(train,np.repeat(train[-1,],step))

trainX,trainY =convertToMatrix(train,step)
testX,testY =convertToMatrix(test,step)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# In[70]:


model_temp = build_simple_rnn(num_units=128,num_dense=32,embedding=8,lr=0.0005)

batch_size=8
num_epochs = 500

model_temp.fit(trainX,trainY, 
          epochs=num_epochs, 
          batch_size=batch_size, 
          callbacks=[MyCallback()],verbose=0)


# In[71]:


plt.figure(figsize=(7,5))
plt.title("RMSE loss over epochs",fontsize=16)
plt.plot(np.sqrt(model_temp.history.history['loss']),c='k',lw=2)
plt.grid(True)
plt.xlabel("Epochs",fontsize=14)
plt.ylabel("Root-mean-squared error",fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# In[72]:


trainPredict = model_temp.predict(trainX)
testPredict= model_temp.predict(testX)
predicted=np.concatenate((trainPredict,testPredict),axis=0)
index = temp_SF.index.values

plt.figure(figsize=(15,5))
plt.title("Temperature: Ground truth and prediction together",fontsize=18)
plt.plot(index,temp_SF['San Francisco'],c='blue')
plt.plot(index,predicted,c='orange',alpha=0.75)
plt.legend(['True data','Predicted'],fontsize=15)
plt.axvline(x=Tp, c="r")
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# In[ ]:


train = np.array(pressure_SF['San Francisco'][:Tp])
test = np.array(pressure_SF['San Francisco'][Tp:])

train=train.reshape(-1,1)
test=test.reshape(-1,1)

step = 8

# add step elements into train and test
test = np.append(test,np.repeat(test[-1,],step))
train = np.append(train,np.repeat(train[-1,],step))

trainX,trainY =convertToMatrix(train,step)
testX,testY =convertToMatrix(test,step)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
model_pressure = build_simple_rnn(num_units=128,num_dense=32,embedding=8,lr=0.0005)

batch_size=8
num_epochs = 500

model_pressure.fit(trainX,trainY, 
          epochs=num_epochs, 
          batch_size=batch_size, 
          callbacks=[MyCallback()],verbose=0)


# In[75]:


plt.figure(figsize=(7,5))
plt.title("RMSE loss over epochs",fontsize=16)
plt.plot(np.sqrt(model_pressure.history.history['loss']),c='k',lw=2)
plt.grid(True)
plt.xlabel("Epochs",fontsize=14)
plt.ylabel("Root-mean-squared error",fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# In[ ]:


trainPredict = model_pressure.predict(trainX)
testPredict= model_pressure.predict(testX)
predicted=np.concatenate((trainPredict,testPredict),axis=0)
index = pressure_SF.index.values

plt.figure(figsize=(15,5))
plt.title("Pressure: Ground truth and prediction together",fontsize=18)
plt.plot(index,pressure_SF['San Francisco'],c='blue')
plt.plot(index,predicted,c='orange',alpha=0.75)
plt.legend(['True data','Predicted'],fontsize=15)
plt.axvline(x=Tp, c="r")
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# In[ ]:




