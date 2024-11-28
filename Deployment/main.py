from flask import Flask, jsonify, request
import pandas as pd
import pytz
import uuid
from datetime import datetime
import numpy as np
from google.cloud import bigquery

app = Flask(__name__)

# Configuration
BUCKET_NAME = "input_data_capstone"
GCS_FILE_PATH = f"gs://{BUCKET_NAME}/input_data/cleaned_data.csv"
BQ_PROJECT = "iitj-capstone-project-group-18"
BQ_DATASET = "CAPSTONE_PROJECT"
BQ_TABLE = "HOTEL_BOOKING"
BQ_PREDICTION_TABLE = "PREDICTION_RESULTS"
BIGQUERY_SCHEMA = {
    'fields': [
        {'name': 'guid', 'type': 'STRING', "mode": "NULLABLE"},
        {'name': 'hotel', 'type': 'STRING', "mode": "NULLABLE"},
        {'name': 'is_canceled', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'lead_time', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'arrival_date_year', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'arrival_date_month', 'type': 'STRING', "mode": "NULLABLE"},
        {'name': 'arrival_date_week_number', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'arrival_date_day_of_month', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'stays_in_weekend_nights', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'stays_in_week_nights', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'adults', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'children', 'type': 'FLOAT', "mode": "NULLABLE"},
        {'name': 'babies', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'meal', 'type': 'STRING', "mode": "NULLABLE"},
        {'name': 'country', 'type': 'STRING', "mode": "NULLABLE"},
        {'name': 'market_segment', 'type': 'STRING', "mode": "NULLABLE"},
        {'name': 'distribution_channel', 'type': 'STRING', "mode": "NULLABLE"},
        {'name': 'is_repeated_guest', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'previous_cancellations', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'previous_bookings_not_canceled', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'reserved_room_type', 'type': 'STRING', "mode": "NULLABLE"},
        {'name': 'assigned_room_type', 'type': 'STRING', "mode": "NULLABLE"},
        {'name': 'booking_changes', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'deposit_type', 'type': 'STRING', "mode": "NULLABLE"},
        {'name': 'agent', 'type': 'STRING', "mode": "NULLABLE"},
        {'name': 'company', 'type': 'STRING', "mode": "NULLABLE"},
        {'name': 'days_in_waiting_list', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'customer_type', 'type': 'STRING', "mode": "NULLABLE"},
        {'name': 'adr', 'type': 'FLOAT', "mode": "NULLABLE"},
        {'name': 'required_car_parking_spaces', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'total_of_special_requests', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'reservation_status', 'type': 'STRING', "mode": "NULLABLE"},
        {'name': 'reservation_status_date', 'type': 'STRING', "mode": "NULLABLE"},
        {'name': 'insert_timestamp', 'type': 'TIMESTAMP', "mode": "NULLABLE"}
    ]
}
PREDICTION_BIGQUERY_SCHEMA = {
    'fields': [
        {'name': 'guid', 'type': 'STRING', "mode": "NULLABLE"},
        {'name': 'lead_time', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'hotel', 'type': 'STRING', "mode": "NULLABLE"},
        {'name': 'market_segment', 'type': 'STRING', "mode": "NULLABLE"},
        {'name': 'previous_cancellations', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'booking_changes', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'total_of_special_requests', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'arrival_date_month', 'type': 'STRING', "mode": "NULLABLE"},
        {'name': 'predicted_is_canceled', 'type': 'BOOLEAN', "mode": "NULLABLE"},
        {'name': 'insert_timestamp', 'type': 'TIMESTAMP', "mode": "NULLABLE"}
    ]
}

def load_data_from_gcs(file_path):
    """Load data from GCS into a pandas DataFrame."""
    return pd.read_csv(file_path)


def generate_synthetic_data(input_df):
    """Generate synthetic data with random sampling and add metadata."""
    # Randomly determine the number of rows to generate (between 0 and 100)
    num_samples = np.random.randint(0, 101)  # Generates a random integer between 0 and 100

    # If num_samples is 0, return None
    if num_samples == 0:
        return None

    synthetic_data = pd.DataFrame()
    for column in input_df.columns:
        synthetic_data[column] = np.random.choice(input_df[column], size=num_samples, replace=True)
    
    # Add GUID and timestamp
    synthetic_data['guid'] = [str(uuid.uuid4()) for _ in range(num_samples)]
    ind_zone = pytz.timezone('Asia/Kolkata')
    synthetic_data['insert_timestamp'] = datetime.now(ind_zone)
    
    # Remove unwanted columns if present
    synthetic_data = synthetic_data.drop('Unnamed: 0', axis=1, errors='ignore')

    return synthetic_data

def generate_predictions(input_df):
    """Generate predictions and add metadata."""
    # Add a synthetic prediction column (replace with actual ML model prediction logic)
    input_df['predicted_is_canceled'] = np.random.choice([True, False], size=len(input_df))

    # Add GUID and timestamp
    input_df['guid'] = [str(uuid.uuid4()) for _ in range(len(input_df))]
    ind_zone = pytz.timezone('Asia/Kolkata')
    input_df['insert_timestamp'] = datetime.now(ind_zone)

    # Select only the columns that match the BigQuery schema
    prediction_columns = [
        "guid", "lead_time", "hotel", "market_segment", 
        "previous_cancellations", "booking_changes", 
        "total_of_special_requests", "arrival_date_month", 
        "predicted_is_canceled", "insert_timestamp"
    ]
    prediction_df = input_df[prediction_columns]
    
    return prediction_df

def write_to_bigquery(df, project_id, dataset_id, table_id, schema):
    """Write the DataFrame to a BigQuery table."""
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    job = client.load_table_from_dataframe(
        dataframe=df,
        destination=table_ref,
        job_config=bigquery.LoadJobConfig(
            schema=schema['fields'],
            write_disposition="WRITE_APPEND",  # Appends to the table
        ),
    )
    job.result()  # Wait for the job to complete
    return f"Data successfully written to {table_ref}."


@app.route('/ingest', methods=['POST'])
def ingest_data():
    """Trigger ingestion pipeline."""
    try:
        # Step 1: Load data from GCS
        input_df = load_data_from_gcs(GCS_FILE_PATH)
        
        # Step 2: Generate synthetic data
        synthetic_df = generate_synthetic_data(input_df)
        
        # Step 3: Handle case where no rows are generated
        if synthetic_df is None:
            return jsonify({"status": "success", "message": "No records to insert (0 rows generated)."}), 200

        # Step 4: Write synthetic data to BigQuery
        result = write_to_bigquery(synthetic_df, BQ_PROJECT, BQ_DATASET, BQ_TABLE, BIGQUERY_SCHEMA)
        return jsonify({"status": "success", "message": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict_data():
    """Trigger prediction pipeline."""
    try:
        # Step 1: Load data from GCS
        input_df = load_data_from_gcs(GCS_FILE_PATH)

        # Step 2: Generate predictions
        prediction_df = generate_predictions(input_df)

        # Step 3: Write predictions to BigQuery
        result = write_to_bigquery(prediction_df, BQ_PROJECT, BQ_DATASET, BQ_PREDICTION_TABLE, PREDICTION_BIGQUERY_SCHEMA)
        return jsonify({"status": "success", "message": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
