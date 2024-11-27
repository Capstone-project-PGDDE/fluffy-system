import pandas as pd
import pytz
import uuid
from datetime import datetime
import numpy as np
from google.cloud import bigquery

# Configuration
BUCKET_NAME = "input_data_capstone"
GCS_FILE_PATH = f"gs://{BUCKET_NAME}/input_data/cleaned_data.csv"
BQ_PROJECT = "iitj-capstone-project-group-18"
BQ_DATASET = "CAPSTONE_PROJECT"
BQ_TABLE = "HOTEL_BOOKING"
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


def load_data_from_gcs(file_path):
    """Load data from GCS into a pandas DataFrame."""
    return pd.read_csv(file_path)


def generate_synthetic_data(input_df, num_samples):
    """Generate synthetic data with random sampling and add metadata."""
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


def write_to_bigquery(df, project_id, dataset_id, table_id):
    """Write the DataFrame to a BigQuery table."""
    client = bigquery.Client(project=project_id)

    # Write to BigQuery
    job = client.load_table_from_dataframe(
        dataframe=df,
        destination=f"{project_id}.{dataset_id}.{table_id}",
        job_config=bigquery.LoadJobConfig(
            schema=BIGQUERY_SCHEMA['fields'],
            write_disposition="WRITE_APPEND",  # Appends to the table
        ),
    )
    job.result()  # Wait for the job to complete
    print(f"Data successfully written to {dataset_id}.{table_id} in BigQuery.")


def main():
    # Step 1: Load data from GCS
    input_df = load_data_from_gcs(GCS_FILE_PATH)
    
    # Step 2: Generate synthetic data
    synthetic_df = generate_synthetic_data(input_df, num_samples=60)
    
    # Step 3: Write the synthetic data to BigQuery
    write_to_bigquery(synthetic_df, BQ_PROJECT, BQ_DATASET, BQ_TABLE)


if __name__ == "__main__":
    main()
