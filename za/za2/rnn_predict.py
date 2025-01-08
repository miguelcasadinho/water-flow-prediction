# Recurrent Neural Network
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all warnings and errors, keep only critical errors

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import pyodbc
from datetime import datetime, timedelta
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore


# Load configuration from environment variables
DATABASE_CONFIG = {
    'host': os.getenv('sqlInoutHost'),
    'database': os.getenv('sqlInoutDatabase'),
    'user': os.getenv('sqlInoutUser'),
    'password': os.getenv('sqlInoutPassword')
}

# Establish connection to the SQL Server database
def connect_to_sql():
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
                Tag_ID = 512
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
    return dataset.reset_index().rename(columns={'index': 'date'}).set_index('date')

#Prepare the input data for model prediction
def prepare_input_for_model(dataset, lookback_period):
    values = dataset['flow'].values
    values = values.reshape(-1, 1)

    # Scale data
    sc = MinMaxScaler(feature_range=(0, 1))
    inputs = sc.fit_transform(values)

    # Create sequences
    X_test = []
    for i in range(lookback_period, len(inputs)):
        X_test.append(inputs[i - lookback_period:i, 0])
    X_test = np.array(X_test)
    return np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)), sc

#Establish connection to the PostgreSQL database.
def connect_to_psql():
    try:
        conn = psycopg2.connect(
            host=os.getenv('psqlGiggoHost'),
            database=os.getenv('psqlGiggoDatabase'),
            user=os.getenv('psqlGiggoUser'),
            password=os.getenv('psqlGiggoPassword2')
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise

#Insert predictions into the database in batches
def insert_predictions(cur, payload):
    try:
        sql = """INSERT INTO flow_predict(device, date, flow) VALUES(%s, %s, %s) ON CONFLICT (device, date) DO NOTHING"""
        cur.executemany(sql, [(item['device'], item['date'], item['flow']) for item in payload])
    except Exception as e:
        print(f"Error inserting predictions: {e}")
        raise

# Main execution
if __name__ == "__main__":
    # Database connection
    conn = connect_to_sql()
    # Fetch and preprocess data
    dataset = fetch_data(conn)
    conn.close()  # Close the connection
    dataset = preprocess_data(dataset)
    
    # Prepare input for model
    lookback_period = 288
    X_test, scaler = prepare_input_for_model(dataset, lookback_period)

    # Load model and make predictions
    model = load_model('/home/giggo/python/water-flow-prediction/za/za2/updated_flow_prediction_model.keras')
    predicted_flow = model.predict(X_test)
    predicted_flow = scaler.inverse_transform(predicted_flow)
    
    # Getting the real flow of the last day
    real_flow = dataset.iloc[289:, 0].values
    #x_value = dataset_test.index.values

    # Prepare the payload for insertion
    payload = []
    device_id = '512'
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    for i in range(len(predicted_flow)):
        payload.append({
            "device": device_id,
            "date": (today + timedelta(minutes=5 * i)).isoformat(),  
            "flow": float(round(predicted_flow[i][0], 3))
        })
    
    # Database connection
    conn = connect_to_psql()
    cur = conn.cursor()

    # Insert predictions into the database
    insert_predictions(cur, payload)

    # Commit and close the database connection
    conn.commit()
    cur.close()
    conn.close()
    print('Predicted flow saved to database')

    # Visualising the results
    plt.plot(predicted_flow, color = 'blue', label = 'Predicted Flow')
    plt.plot(real_flow, color = 'red', label = 'Real Flow')
    plt.title('ZMC ZA2 - Flow Prediction')
    plt.xlabel('Time (5-min intervals)')
    plt.ylabel('Flow (m3/h)')
    plt.legend()
    #plt.xticks(rotation=45)
    plt.tight_layout() 
    plt.show()
    plt.savefig('/home/giggo/python/water-flow-prediction/za/za2/predict.png')
