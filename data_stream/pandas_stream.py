import pandas as pd
import json
import uuid
import pytz
from datetime import datetime 
from kafka import KafkaConsumer
from google.cloud import bigquery
import logging

# Set logging levels to suppress unnecessary logs but keep important ones
logging.getLogger("kafka").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)

# Define the schema with the new 'guid' column
columns = [
    "hotel", "is_canceled", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_week_number", "arrival_date_day_of_month", "stays_in_weekend_nights", 
    "stays_in_week_nights", "adults", "children", "babies", "meal", "country", 
    "market_segment", "distribution_channel", "is_repeated_guest", "previous_cancellations", 
    "previous_bookings_not_canceled", "reserved_room_type", "assigned_room_type", 
    "booking_changes", "deposit_type", "agent", "company", "days_in_waiting_list", 
    "customer_type", "adr", "required_car_parking_spaces", "total_of_special_requests", 
    "reservation_status", "reservation_status_date", "insert_timestamp","guid"  
]

# Kafka consumer setup
consumer = KafkaConsumer(
    'hotel-bookings',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

# BigQuery client setup
client = bigquery.Client()
table_id = 'capstone-project-group-18.Capstone_Source.hotel_bookings_stream'

# Function to write to BigQuery
def write_to_bigquery(df):
    df.to_gbq(destination_table=table_id, project_id='capstone-project-group-18', if_exists='append')

# Monitor Kafka and insert data with GUID
for message in consumer:
    data = message.value
    guid = str(uuid.uuid4())  # Generate a unique GUID as a string
    data['guid'] = guid  # Add the GUID to the data
    UTC = pytz.utc 
    ind_zone = pytz.timezone('Asia/Kolkata')
    data['insert_timestamp'] = datetime.now(ind_zone)  # Add current timestamp in UTC
    df = pd.DataFrame([data], columns=columns)   
    logging.info(f"Inserting record with GUID: {guid}")
    write_to_bigquery(df)
