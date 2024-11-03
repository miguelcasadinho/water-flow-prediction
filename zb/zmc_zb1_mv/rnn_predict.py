# Recurrent Neural Network
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress all warnings and errors, keep only critical errors

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import logging
from datetime import datetime, timedelta
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore


# Set logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#Establish connection to the PostgreSQL database.
def connect_to_db():
    try:
        conn = psycopg2.connect(
            host=os.getenv('psqlGiggoHost'),
            database=os.getenv('psqlGiggoDatabase'),
            user=os.getenv('psqlGiggoUser'),
            password=os.getenv('psqlGiggoPassword')
        )
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
    dataset['flow'] = dataset['flow'].astype(float)  # Ensure 'flow' is of type float
    dataset['flow'] = dataset['flow'].ffill().interpolate(method='linear')  # Fill missing values

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

#Insert predictions into the database in batches
def insert_predictions(cur, payload):
    try:
        sql = """INSERT INTO flow_predict(device, date, flow) VALUES(%s, %s, %s) ON CONFLICT (device, date) DO NOTHING"""
        cur.executemany(sql, [(item['device'], item['date'], item['flow']) for item in payload])
    except Exception as e:
        logging.error(f"Error inserting predictions: {e}")
        raise

# Main execution
if __name__ == "__main__":
    # Database connection
    conn = connect_to_db()
    cur = conn.cursor()

    # Fetch and preprocess data
    dataset = fetch_data(cur)
    dataset = preprocess_data(dataset)

    # Prepare input for model
    lookback_period = 96
    X_test, scaler = prepare_input_for_model(dataset, lookback_period)

    # Load model and make predictions
    model = load_model('./updated_flow_prediction_model.keras')
    predicted_flow = model.predict(X_test)
    predicted_flow = scaler.inverse_transform(predicted_flow)

    # Prepare the payload for insertion
    payload = []
    device_id = '202035132'
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    for i in range(len(predicted_flow)):
        payload.append({
            "device": device_id,
            "date": (today + timedelta(minutes=15 * i)).isoformat(),  
            "flow": float(round(predicted_flow[i][0], 3))
        })

    # Insert predictions into the database
    insert_predictions(cur, payload)

    # Commit and close the database connection
    conn.commit()
    cur.close()
    conn.close()

    # Visualising the results
    plt.plot(predicted_flow, color = 'blue', label = 'Predicted Flow')
    plt.title('ZMC Moinhos Velhos - Flow Prediction')
    plt.xlabel('Time (15-min intervals)')
    plt.ylabel('Flow (m3/h)')
    plt.legend()
    #plt.xticks(rotation=45)
    plt.tight_layout() 
    plt.show()
    plt.savefig('predict.png')

