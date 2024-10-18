import pandas as pd
import numpy as np
import time
import json
import random

# Load the dataset (for demo purposes)
input_data = pd.read_csv('data/hotel_bookings.csv')

# Data Cleaning Steps
# Fill missing values in 'children' column with the median value
input_data['children'].fillna(input_data['children'].median(), inplace=True)

# Fill missing values in 'country' column with the most frequent value
input_data['country'].fillna(input_data['country'].mode()[0], inplace=True)

# Cast 'agent' and 'company' columns to 'object' (string) type and fill missing values with 'Unknown'
input_data['agent'] = input_data['agent'].astype('object').fillna('Unknown')
input_data['company'] = input_data['company'].astype('object').fillna('Unknown')

# Function to create synthesized data by adding random noise to numerical columns
def generate_synthesized_data(row):
    """
    Generates synthesized data by adding random noise to numeric columns.

    Parameters:
    - row: pd.Series, a single row of the original dataset.

    Returns:
    - pd.Series with synthesized data.
    """
    # Copy the original row to avoid modifying the original dataset
    synthetic_row = row.copy()
    
    # Add noise to numeric columns (small random noise proportional to each column's std)
    numeric_columns = ['lead_time', 'adr', 'days_in_waiting_list', 'adults', 'children', 'babies']
    
    for col in numeric_columns:
        if col in synthetic_row:
            noise = np.random.normal(0, 0.05 * row[col])  # Add 5% noise
            synthetic_row[col] += noise
    
    return synthetic_row

# Function to simulate live data stream, sending both original and synthesized data
def simulate_live_data(data, interval=1, synth_prob=0.5):
    """
    Simulates real-time data stream from a dataset, sending both original and synthesized data.

    Parameters:
    - data: pd.DataFrame, the dataset to stream from.
    - interval: int, the time interval (in seconds) between each record sent (simulates real-time).
    - synth_prob: float, the probability of sending synthesized data (0 <= synth_prob <= 1).
    """
    for i, row in data.iterrows():
        # Randomly choose whether to send original or synthesized data
        if random.random() < synth_prob:
            # Generate synthesized data
            record = generate_synthesized_data(row).to_dict()
            data_type = "synthesized"
        else:
            # Send original data
            record = row.to_dict()
            data_type = "original"

        # Convert the row to a JSON object for live streaming
        live_data = json.dumps(record)

        # Simulate sending the live data (for demo, we'll just print it)
        print(f"Sending {data_type} data record {i+1}: {live_data}")

        # Wait for the specified interval to simulate real-time streaming
        time.sleep(interval)

# Simulate live streaming of the first 10 rows with 2-second intervals
# 50% chance of sending synthesized data (synth_prob=0.5)
simulate_live_data(input_data.head(10), interval=2, synth_prob=0.5)
