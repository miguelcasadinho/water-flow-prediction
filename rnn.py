# Recurrent Neural Network
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses all logs


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
# Read the csv with pandas
dataset_train = pd.read_csv(('train_set.csv'), delimiter=';')
# Leave only the flow column
training_set = dataset_train.iloc[:, 3:4].values

# Getting the real flow of the last day
dataset_test = pd.read_csv('test_set.csv')
real_flow = dataset_test.iloc[:, 3:4].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
# Scale the flow values from 0 to 1
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 1344 timesteps and 1 output
X_train = []
y_train = []
# 7 days are 672 (4x24x7) and 13094 all the records in the training set 
for i in range(672, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-672:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
# Create a 3º dimension
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential # type: ignore
from keras.layers import Dense # type: ignore
from keras.layers import LSTM # type: ignore
from keras.layers import Dropout # type: ignore

# Initialising the RNN
regressor = Sequential()

# Create 50 neurons and drop 20%
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))



# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Callbacks for early stopping and model checkpointing
from keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
early_stopping = EarlyStopping(monitor='loss', patience=10)
model_checkpoint = ModelCheckpoint('best_flow_prediction_model.keras', save_best_only=True)

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 50, batch_size = 32, callbacks=[early_stopping, model_checkpoint])

# Save the model
regressor.save('flow_prediction_model.keras')  # Save the model

# Part 3 - Making the predictions and visualising the results

# Getting the real flow of the last day
dataset_test = pd.read_csv(('test_set.csv'), delimiter=';')
real_flow = dataset_test.iloc[:, 3:4].values


# Getting the predicted flow of the last day
dataset_total = pd.concat((dataset_train['flow'], dataset_test['flow']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 672:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(672, 768):
    X_test.append(inputs[i-672:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_flow = regressor.predict(X_test)
predicted_flow = sc.inverse_transform(predicted_flow)

# Visualising the results
plt.plot(real_flow, color = 'red', label = 'Real Flow')
plt.plot(predicted_flow, color = 'blue', label = 'Predicted Flow')
plt.title('ZMC Moinhos de Santa Maria - Flow Prediction')
plt.xlabel('Time')
plt.ylabel('Flow (m3/h)')
plt.legend()

# Save the plot to a file instead of displaying it
plt.savefig('flow_prediction.png')

plt.show()


