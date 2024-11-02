# Recurrent Neural Network
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses all logs

# Part 1 - Data cleaning

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import the dataset
dataset = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
#print(dataset_test)

# Arredondar para o múltiplo de 15 minutos mais próximo
def round_to_nearest_15_minutes(timestamp):
    minutes = timestamp.minute
    nearest_15 = 15 * round(minutes / 15)

    if nearest_15 == 60:
        # Verifique se a hora é 23 e, se for, ajuste para o próximo dia
        if timestamp.hour == 23:
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            return timestamp.replace(hour=timestamp.hour + 1, minute=0, second=0, microsecond=0)
    else:
        return timestamp.replace(minute=nearest_15, second=0, microsecond=0)

# Convert 'Time' column to datetime objects
dataset['Time'] = pd.to_datetime(dataset['Time'])
dataset_test['Time'] = pd.to_datetime(dataset_test['Time'])
dataset['Time'] = dataset['Time'].apply(round_to_nearest_15_minutes)
dataset_test['Time'] = dataset_test['Time'].apply(round_to_nearest_15_minutes)
#print(dataset_test)

# Handling rows with NaN values
# Identify rows with NaN values
dataset_rows_with_nan = dataset[dataset.isnull().any(axis=1)]
#print("Rows with NaN in dataset:", dataset_rows_with_nan)
dataset_test_rows_with_nan = dataset_test[dataset_test.isnull().any(axis=1)]
#print("Rows with NaN in dataset:", dataset_test_rows_with_nan)
# Deleting rows with NaN values
dataset.dropna(inplace=True)
dataset_test.dropna(inplace=True)

#Handling duplicates timestamps
dataset_duplicate_rows = dataset[dataset.duplicated(subset=['Time'])]
#print("Number of duplicate rows in dataset:", len(dataset_duplicate_rows))
dataset_test_duplicate_rows = dataset_test[dataset_test.duplicated(subset=['Time'])]
#print("Number of duplicate rows in dataset_test:", len(dataset_test_duplicate_rows))
# Remove the duplicate rows based on Time:
dataset.drop_duplicates(subset=['Time'], inplace=True)
dataset_test.drop_duplicates(subset=['Time'], inplace=True)

#Outlier detection and treatment using Z-score
from scipy.stats import zscore
# Calculate Z-scores for the dataset
z_scores_dataset = zscore(dataset['flow'])
z_scores_dataset_test = zscore(dataset_test['flow'])
# Set the Z-score threshold (e.g., 5 standard deviations)
threshold = 5
# Identify outliers based on Z-score threshold
dataset_outliers =  np.where(np.abs(z_scores_dataset) > threshold)
#print("Outliers identified in dataset:", dataset.iloc[dataset_outliers[0]])
dataset_test_outliers =  np.where(np.abs(z_scores_dataset_test) > threshold)
#print("Outliers identified in dataset_test:", dataset_test.iloc[dataset_test_outliers[0]])
# Remove outliers from the dataset
dataset = dataset[(np.abs(z_scores_dataset) <= threshold)]
dataset_test = dataset_test[(np.abs(z_scores_dataset_test) <= threshold)]

# Find gaps in the timedata
# Create a complete time index from min to max with 5-minute frequency
dataset_complete_time_index = pd.date_range(start=dataset['Time'].min(), end=dataset['Time'].max(), freq='15min')
dataset_test_complete_time_index = pd.date_range(start=dataset_test['Time'].min(), end=dataset_test['Time'].max(), freq='15min')
# Reindex the dataset with the complete time index
dataset = dataset.set_index('Time').reindex(dataset_complete_time_index)
dataset_test = dataset_test.set_index('Time').reindex(dataset_test_complete_time_index)
# Reset index to bring 'Time' back as a column
dataset.reset_index(inplace=True)
dataset_test.reset_index(inplace=True)
dataset.rename(columns={'index': 'Time'}, inplace=True)
dataset_test.rename(columns={'index': 'Time'}, inplace=True)
# Fill the NaN values in the 'Distribuído' column using forward fill and linear interpolation
dataset['flow'] = dataset['flow'].ffill()  # Forward fill
dataset['flow'] = dataset['flow'].interpolate(method='linear')  # Linear interpolation
dataset_test['flow'] = dataset_test['flow'].ffill()  # Forward fill
dataset_test['flow'] = dataset_test['flow'].interpolate(method='linear')  # Linear interpolation

#Set time as index and extract 'flow'
dataset.set_index('Time', inplace=True)
dataset_test.set_index('Time', inplace=True)
values = dataset['flow'].values
values = dataset_test['flow'].values

#Feature engineering (extract useful time features)
dataset['Month'] = dataset.index.month
dataset_test['Month'] = dataset_test.index.month
dataset['day_of_month'] = dataset.index.day
dataset_test['day_of_month'] = dataset_test.index.day
dataset['day_of_week'] = dataset.index.dayofweek
dataset_test['day_of_week'] = dataset_test.index.dayofweek
dataset['Hour'] = dataset.index.hour
dataset_test['Hour'] = dataset_test.index.hour
dataset['Minute'] = dataset.index.minute
dataset_test['Minute'] = dataset_test.index.minute

#Save to CSV file
dataset.to_csv('dataset.csv', index=True)
dataset_test.to_csv('dataset_test.csv', index=True)


# Part 2 - Data preprocessing

# Leave only the flow column
training_set = dataset.iloc[:, 0:1].values
#print(training_set)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
#print(training_set_scaled)

# Creating a data structure with 1 day timesteps and 1 output
X_train = []
y_train = []
# 1 day are 96 (4x24) sequences
sequence_length = 96
for i in range(sequence_length, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-sequence_length:i])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
#print(X_train[0])
#print(y_train)

# Reshaping input data for LSTM: [samples, time steps, features]
#print(X_train.shape)
#print(y_train.shape)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
print(X_train.shape)


# Part 3 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Create 50 neurons and drop 20%
# Add an Input layer to specify the shape of the input
regressor.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
# Adding the LSTM layers and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2))
# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
from keras.metrics import MeanAbsoluteError # Monitoring Model Performance
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=[MeanAbsoluteError()])

# Callbacks for early stopping and model checkpointing
from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint('best_flow_prediction_model.keras', save_best_only=True)

# Validation Set
from sklearn.model_selection import train_test_split
# Splitting the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 15, batch_size = 32,
              validation_data=(X_val, y_val),
              callbacks=[early_stopping, model_checkpoint])

# Save the model
regressor.save('flow_prediction_model.keras')


# Part 4 - Making the predictions and visualising the results

# Getting the real flow of the last day
real_flow = dataset_test.iloc[:, 0].values
#x_value = dataset_test.index.values

# Getting the predicted flow of the last day
dataset_total = pd.concat((dataset['flow'], dataset_test['flow']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - len(dataset_test):].values
inputs = inputs.reshape(-1,1)
inputs = sc.fit_transform(inputs)

# Define the lookback period for sequence input to the model
lookback_period = len(dataset_test)

X_test = []
for i in range(lookback_period, len(inputs)):
    X_test.append(inputs[i - lookback_period:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions using the trained model
predicted_flow = regressor.predict(X_test)
predicted_flow = sc.inverse_transform(predicted_flow)

# Visualising the results
plt.plot(real_flow, color = 'red', label = 'Real Flow')
plt.plot(predicted_flow, color = 'blue', label = 'Predicted Flow')
plt.title('ZMC Moinhos Velhos - Flow Prediction')
plt.xlabel('Time')
plt.ylabel('Flow (m3/h)')
plt.legend()
# Save the plot to a file 
plt.savefig('flow_prediction.png')
plt.show()