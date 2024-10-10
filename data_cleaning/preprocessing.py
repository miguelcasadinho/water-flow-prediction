# Data Preprocessing 

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('zb2_jan2set.csv', parse_dates=['Time'])

# Set time as index and extract 'flow'
dataset.set_index('Time', inplace=True)
values = dataset['Distribuído'].values

# Feature engineering (extract useful time features)
dataset['Year'] = dataset.index.year
dataset['Month'] = dataset.index.month
dataset['day_of_month'] = dataset.index.day
dataset['day_of_week'] = dataset.index.dayofweek
dataset['Hour'] = dataset.index.hour
dataset['Minute'] = dataset.index.minute
dataset['Second'] = dataset.index.second


# Identify rows with NaN values

#rows_with_nan = dataset[dataset.isnull().any(axis=1)] 
#print(rows_with_nan)

# Deleting rows with NaN values 
#dataset.dropna(inplace=True)

# Replace NaN values with the mean of the column
#dataset.fillna(dataset.mean(), inplace=True)

# Replace NaN values with the mean of the column with sklearn library
#from sklearn.impute import SimpleImputer
# Create an imputer object with a strategy to replace NaN values with the mean
#imputer = SimpleImputer(strategy='mean')
# Apply the imputer on the dataset, converting it back to a DataFrame with the same columns
#dataset_imputed = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)


# Outlier detection and treatment

# Using Z-score
#from scipy.stats import zscore
# Calculate Z-scores for the dataset
#z_scores = zscore(dataset)
# Set the Z-score threshold (e.g., 3 standard deviations)
#threshold = 3
# Identify outliers based on Z-score threshold
#outliers = np.where(np.abs(z_scores) > threshold)
# Print the outliers
#print("Outliers identified using Z-score method:")
#print(dataset.iloc[outliers[0]])

# Using quartiles
# Calculate quartiles and IQR
#q1 = np.percentile(dataset, 25, axis=0)  # First quartile (25th percentile)
#q3 = np.percentile(dataset, 75, axis=0)  # Third quartile (75th percentile)
#iqr = q3 - q1  # Interquartile range

# Identify outliers based on IQR
#lower_bound = q1 - (1.5 * iqr)
#upper_bound = q3 + (1.5 * iqr)

# Identify outliers
#outliers = np.where((dataset < lower_bound) | (dataset > upper_bound))

# Print the outliers
#print("Outliers identified using IQR method:")
#print(dataset.iloc[outliers[0]])

# Remove outliers from the dataset
#dataset = dataset[(np.abs(z_scores) <= threshold)]
#dataset = dataset[(dataset >= lower_bound) | (dataset <= upper_bound)]

# Replace outliers with the mean value
#dataset[outliers] = dataset.mean()


# Dealing with duplicates
duplicate_rows = dataset[dataset.duplicated()]

# Print the number of duplicate rows
#print("Number of duplicate rows:", len(duplicate_rows))

# Print the duplicate rows themselves
#print("Duplicate rows identified:")
#print(duplicate_rows)

# Remove the duplicate rows from the dataset:
dataset.drop_duplicates(inplace=True)


#Finding gaps in the timedata

# Calculate the difference between consecutive rows
#dataset['time_diff'] = dataset.index.to_series().diff()

# Define the threshold for gaps (e.g., greater than 10 minutes)
#gap_threshold = pd.Timedelta('10min')
#gaps = dataset[dataset['time_diff'] > gap_threshold]
#print(gaps)



'''
# Dataset Profiling Report

# Importing the necessary library for profiling
from ydata_profiling import ProfileReport

# Generate the profile report
profile = ProfileReport(dataset, title="Dataset Profiling Report", explorative=True)

# Save the report to an HTML file
profile.to_file("dataset_profile_report.html")

# Optionally display the report within a Jupyter Notebook (if running there)
# profile.to_notebook_iframe()
'''

