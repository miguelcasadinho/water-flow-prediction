# Recurrent Neural Network
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all warnings and errors, keep only critical errors

# Import necessary libraries
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pyodbc
from scipy.stats import zscore
from keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# Load configuration from environment variables
DATABASE_CONFIG = {
    'host': os.getenv('sqlInoutHost'),
    'database': os.getenv('sqlInoutDatabase'),
    'user': os.getenv('sqlInoutUser'),
    'password': os.getenv('sqlInoutPassword')
}

LOOKBACK_PERIOD = 288
MODEL_PATH = '/home/giggo/python/water-flow-prediction/za/za3/updated_flow_prediction_model.keras'
UPDATED_MODEL_PATH = '/home/giggo/python/water-flow-prediction/za/za3/updated_flow_prediction_model.keras'
timestamp = datetime.now().strftime("%Y:%m:%d_%H:%M")
UPDATED_MODEL_PATH_VERSION = f'/home/giggo/python/water-flow-prediction/za/za3/models/model_{timestamp}.keras'

# Establish connection to the SQL Server database
def connect_to_db():
    try:
        connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={DATABASE_CONFIG['host']};"
            f"DATABASE={DATABASE_CONFIG['database']};"
            f"UID={DATABASE_CONFIG['user']};"
            f"PWD={DATABASE_CONFIG['password']}"
        )
        conn = pyodbc.connect(connection_string)
        print("Database connection established successfully.")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise

# Fetch flow data from the database
def fetch_data(conn):
    try:
        query = """
            SELECT 
                Data as date,
                Valor as flow
            FROM
                Go_Ready.dbo.Telegestao_data
            WHERE
                Tag_ID = 536
                AND Data BETWEEN 
                    CAST(DATEADD(day, -2, GETDATE()) AS DATE) -- 00:00 of 2 days ago
                    AND CAST(GETDATE() AS DATE)  -- 00:00 of today
            ORDER BY Data ASC
        """
        # Create a cursor from the connection
        cur = conn.cursor()
        
        # Execute the query
        cur.execute(query)
        
        # Fetch all rows
        rows = cur.fetchall()

        # Convert rows to a list of tuples with floats instead of Decimals
        rows = [(date, float(flow)) for date, flow in rows]

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=["date", "flow"])
        
        print("Data fetched successfully from the database.")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise
    finally:
        cur.close()  # Close the cursor

# Preprocess the dataset: round timestamps, handle NaNs, remove duplicates and outliers
def preprocess_data(dataset):
    # Round to nearest 5 minutes
    def round_to_nearest_5_minutes(timestamp):
        minutes = timestamp.minute
        nearest_5 = 5 * round(minutes / 5)
        if nearest_5 == 60:
            return timestamp.replace(hour=timestamp.hour + 1, minute=0, second=0, microsecond=0) if timestamp.hour < 23 else timestamp.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return timestamp.replace(minute=nearest_5, second=0, microsecond=0)

    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset['date'] = dataset['date'].apply(round_to_nearest_5_minutes)

    # Handle NaNs
    dataset.dropna(inplace=True)

    # Remove duplicates
    dataset.drop_duplicates(subset=['date'], inplace=True)
    dataset_size = len(dataset)

    # Outlier detection using Z-score
    z_scores = zscore(dataset['flow'])
    dataset = dataset[(np.abs(z_scores) <= 5)]  # Remove outliers

    # Complete time index and fill missing values
    complete_time_index = pd.date_range(start=dataset['date'].min(), end=dataset['date'].max(), freq='5min')
    dataset = dataset.set_index('date').reindex(complete_time_index)

    # Ensure the 'flow' column is numeric and fill missing values
    dataset['flow'] = dataset['flow'].astype(float)
    dataset['flow'] = dataset['flow'].ffill().interpolate(method='linear')

    print("Data preprocessing completed.")
    return dataset.reset_index().rename(columns={'index': 'date'}).set_index('date'), dataset_size

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
    print("Input prepared with new shape")
    
    return X_new, y_new, sc  # Return scaler for inverse transformation later

# Function to evaluate the model
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"Model evaluation completed. Mean Squared Error: {mse:.4f}")

# Function to update the model with new data
def update_model_with_new_data(model, X_new, y_new, epochs=5, batch_size=32):
    model.fit(X_new, y_new, epochs=epochs, batch_size=batch_size, verbose=1)  # Use fewer epochs to fine-tune
    print("Model updated with new data.")
    return model

# Function to load new incoming data and updating the model
def dynamic_model_update():
    # Database connection
    conn = connect_to_db()

    # Fetch and preprocess data
    dataset = fetch_data(conn)
    conn.close()  # Close the connection
    dataset, dataset_size = preprocess_data(dataset)

    # Check if the dataset length is greater than 550 after preprocessing
    if dataset_size > 550:
        print(f"Dataset length after preprocessing is {dataset_size}. Proceeding with model update.")

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
        updated_model.save(UPDATED_MODEL_PATH_VERSION)
        print("Model updated and saved successfully.")
    else:
        print(f"Dataset length after preprocessing is {dataset_size}, which is less than 550. Model update skipped.")

# Run the dynamic model update process
if __name__ == "__main__":
    try:
        dynamic_model_update()
    except Exception as e:
        print(f"An error occurred during model update: {e}")