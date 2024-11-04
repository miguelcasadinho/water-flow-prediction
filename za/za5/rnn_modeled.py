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
dataset = pd.read_csv('za5_train.csv')
dataset_test = pd.read_csv('za5_test.csv')
#print(dataset_test)

# Arredondar para o múltiplo de 5 minutos mais próximo
def round_to_nearest_5_minutes(timestamp):
    minutes = timestamp.minute
    nearest_5 = 5 * round(minutes / 5)

    if nearest_5 == 60:
        # Verifique se a hora é 23 e, se for, ajuste para o próximo dia
        if timestamp.hour == 23:
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            return timestamp.replace(hour=timestamp.hour + 1, minute=0, second=0, microsecond=0)
    else:
        return timestamp.replace(minute=nearest_5, second=0, microsecond=0)

# Convert 'Time' column to datetime objects
dataset['Time'] = pd.to_datetime(dataset['Time'])
dataset_test['Time'] = pd.to_datetime(dataset_test['Time'])
dataset['Time'] = dataset['Time'].apply(round_to_nearest_5_minutes)
dataset_test['Time'] = dataset_test['Time'].apply(round_to_nearest_5_minutes)
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
z_scores_dataset = zscore(dataset['Distribuído'])
z_scores_dataset_test = zscore(dataset_test['Distribuído'])
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
dataset_complete_time_index = pd.date_range(start=dataset['Time'].min(), end=dataset['Time'].max(), freq='5min')
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
dataset['Distribuído'] = dataset['Distribuído'].ffill()  # Forward fill
dataset['Distribuído'] = dataset['Distribuído'].interpolate(method='linear')  # Linear interpolation
dataset_test['Distribuído'] = dataset_test['Distribuído'].ffill()  # Forward fill
dataset_test['Distribuído'] = dataset_test['Distribuído'].interpolate(method='linear')  # Linear interpolation

#Set time as index and extract 'flow'
dataset.set_index('Time', inplace=True)
dataset_test.set_index('Time', inplace=True)
values = dataset['Distribuído'].values
values = dataset_test['Distribuído'].values

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


# Part 2 - Making the predictions and visualising the results

# Load the saved model
from keras.models import load_model
model = load_model('./best_flow_prediction_model.keras')
#print(model.summary())

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

# Getting the real flow of the last day
real_flow = dataset_test.iloc[:, 0].values
#x_value = dataset_test.index.values

# Getting the predicted flow of the last day
dataset_total = pd.concat((dataset['Distribuído'], dataset_test['Distribuído']), axis = 0)
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
predicted_flow = model.predict(X_test)
predicted_flow = sc.inverse_transform(predicted_flow)

# Visualising the results
plt.plot(real_flow, color = 'red', label = 'Real Flow')
plt.plot(predicted_flow, color = 'blue', label = 'Predicted Flow')
plt.title('ZMC ZA2 - Flow Prediction')
plt.xlabel('Time')
plt.ylabel('Flow (m3/h)')
plt.legend()
# Save the plot to a file 
plt.savefig('flow_prediction_modeled.png')
plt.show()