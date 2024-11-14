import pandas as pd
import numpy as np
from kafka import KafkaProducer
import json
import time
import os

df = pd.read_csv('./data/cleaned_data.csv')
original_data = df.copy()

# Number of synthetic samples to generate (100 rows per hour)
num_synthetic_samples = 60

# Function to generate synthetic data
def generate_synthetic_data():
    synthetic_data = pd.DataFrame()
    for column in original_data.columns:
        synthetic_data[column] = np.random.choice(original_data[column], size=num_synthetic_samples, replace=True)
    return synthetic_data

# Setting up Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Function to send synthetic data to Kafka
def send_data_to_kafka(data, topic="hotel-bookings"):
    for index, row in data.iterrows():
        try:
            # Convert each row to a dictionary and send to Kafka
            producer.send(topic, row.to_dict())
            print(f"Sent row {index} to Kafka")
        except Exception as e:
            print(f"Error sending data to Kafka: {e}")
        time.sleep(36)  # Sleep for 36 seconds to insert 100 rows per hour

# Run the infinite loop to generate and stream data
try:
    while True:
        synthetic_data = generate_synthetic_data()  # Generate 60 rows of synthetic data
        send_data_to_kafka(synthetic_data)
        producer.flush()  # Ensure all messages are sent
except KeyboardInterrupt:
    print("Streaming stopped by user")
finally:
    producer.close()  
