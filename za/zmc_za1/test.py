# Import necessary libraries
import schedule
import time
import subprocess
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pyodbc
from scipy.stats import zscore


# Set logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Load configuration from environment variables
DATABASE_CONFIG = {
    'host': os.getenv('sqlInoutHost'),
    'database': os.getenv('sqlInoutDatabase'),
    'user': os.getenv('sqlInoutUser'),
    'password': os.getenv('sqlInoutPassword')
}


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
        logging.info("Database connection established successfully.")
        return conn
    except Exception as e:
        logging.error(f"Error connecting to database: {e}")
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
                Tag_ID = 37
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
        
        logging.info("Data fetched successfully from the database.")
        return df
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
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

    logging.info("Data preprocessing completed.")
    return dataset.reset_index().rename(columns={'index': 'date'}).set_index('date')

def run_script():
    subprocess.call(['/path/to/your/venv/bin/python', '/path/to/your/script.py'])
    # Database connection
    conn = connect_to_db()

    # Fetch and preprocess data
    dataset = fetch_data(conn)
    conn.close()  # Close the connection
    dataset = preprocess_data(dataset)
    print(dataset)

# Schedule the job 
schedule.every().day.at("08:00").do(run_script)

# Infinite loop to keep the script running
while True:
    schedule.run_pending()
    time.sleep(1)