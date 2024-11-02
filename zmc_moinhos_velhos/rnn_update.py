# Recurrent Neural Network
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress all warnings and errors, keep only critical errors

# Import necessary libraries
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import psycopg2
from scipy.stats import zscore
from keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Set logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration from environment variables
DATABASE_CONFIG = {
    'host': os.getenv('psqlGiggoHost'),
    'database': os.getenv('psqlGiggoDatabase'),
    'user': os.getenv('psqlGiggoUser'),
    'password': os.getenv('psqlGiggoPassword')
}

LOOKBACK_PERIOD = 96
MODEL_PATH = './flow_prediction_model.keras'
UPDATED_MODEL_PATH = 'updated_flow_prediction_model.keras'

# Establish connection to the PostgreSQL database.
def connect_to_db():
    try:
        conn = psycopg2.connect(**DATABASE_CONFIG)
        logging.info("Database connection established successfully.")
        return conn
    except Exception as e:
        logging.error(f"Error connecting to database: {e}")
        raise

# Fetch flow data from the database
def fetch_data(cur):
    try:
        cur.execute("""
            SELECT 
                date,
                flow
            FROM
                flow
            WHERE
                device = '202035132'
                AND date BETWEEN (CURRENT_DATE - interval '2 days') AND (CURRENT_DATE)
            ORDER BY date ASC
        """)
        rows = cur.fetchall()
        logging.info("Data fetched successfully from the database.")
        return pd.DataFrame(rows, columns=["date", "flow"])
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        raise

# Preprocess the dataset: round timestamps, handle NaNs, remove duplicates and outliers
def preprocess_data(dataset):
    # Round to nearest 15 minutes
    def round_to_nearest_15_minutes(timestamp):
        minutes = timestamp.minute
        nearest_15 = 15 * round(minutes / 15)
        if nearest_15 == 60:
            return timestamp.replace(hour=timestamp.hour + 1, minute=0, second=0, microsecond=0) if timestamp.hour < 23 else timestamp.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return timestamp.replace(minute=nearest_15, second=0, microsecond=0)

    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset['date'] = dataset['date'].apply(round_to_nearest_15_minutes)

    # Handle NaNs
    dataset.dropna(inplace=True)

    # Remove duplicates
    dataset.drop_duplicates(subset=['date'], inplace=True)

    # Outlier detection using Z-score
    z_scores = zscore(dataset['flow'])
    dataset = dataset[(np.abs(z_scores) <= 5)]  # Remove outliers

    # Complete time index and fill missing values
    complete_time_index = pd.date_range(start=dataset['date'].min(), end=dataset['date'].max(), freq='15min')
    dataset = dataset.set_index('date').reindex(complete_time_index)

    # Ensure the 'flow' column is numeric and fill missing values
    dataset['flow'] = dataset['flow'].astype(float)
    dataset['flow'] = dataset['flow'].ffill().interpolate(method='linear')

    logging.info("Data preprocessing completed.")
    return dataset.reset_index().rename(columns={'index': 'date'}).set_index('date')

# Prepare the input data for the model 
def prepare_input_for_model(dataset, lookback_period):
    values = dataset['flow'].values.reshape(-1, 1)

    # Scale data
    sc = MinMaxScaler(feature_range=(0, 1))
    inputs = sc.fit_transform(values)

    # Create sequences and targets
    X_new, y_new = [], []
    for i in range(lookback_period, len(inputs)):
        X_new.append(inputs[i - lookback_period:i, 0])
        y_new.append(inputs[i, 0])
        
    X_new, y_new = np.array(X_new), np.array(y_new)
    X_new = np.reshape(X_new, (X_new.shape[0], X_new.shape[1], 1))  # Reshape for LSTM input
    logging.info(f"Input prepared with shape: {X_new.shape}")
    
    return X_new, y_new, sc  # Return scaler for inverse transformation later

# Function to evaluate the model
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    logging.info(f"Model evaluation completed. Mean Squared Error: {mse:.4f}")

# Function to update the model with new data
def update_model_with_new_data(model, X_new, y_new, epochs=5, batch_size=32):
    model.fit(X_new, y_new, epochs=epochs, batch_size=batch_size, verbose=1)  # Use fewer epochs to fine-tune
    logging.info("Model updated with new data.")
    return model

# Function to load new incoming data and updating the model
def dynamic_model_update():
    # Database connection
    conn = connect_to_db()
    cur = conn.cursor()

    # Fetch and preprocess data
    dataset = fetch_data(cur)
    dataset = preprocess_data(dataset)

    # Prepare input for model
    X_new, y_new, scaler = prepare_input_for_model(dataset, LOOKBACK_PERIOD)
    
    # Load model 
    model = load_model(MODEL_PATH)
    
    # Evaluate the model before updating
    evaluate_model(model, X_new, y_new)
    
    # Update the model with the new data
    updated_model = update_model_with_new_data(model, X_new, y_new)
    
    # Save the updated model 
    updated_model.save(UPDATED_MODEL_PATH)
    logging.info("Model updated and saved successfully.")

# Run the dynamic model update process
if __name__ == "__main__":
    try:
        dynamic_model_update()
    except Exception as e:
        logging.error(f"An error occurred during model update: {e}")


