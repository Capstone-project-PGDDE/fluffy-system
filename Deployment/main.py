from flask import Flask, jsonify, request
import pandas as pd
import pytz
import uuid
import sys
from google.cloud import storage
import joblib
from datetime import datetime
import numpy as np
from google.cloud import bigquery
from sklearn.preprocessing import LabelEncoder
import logging
app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("prediction_log.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)
# Configuration
BUCKET_NAME = "input_data_capstone"
GCS_FILE_PATH = f"gs://{BUCKET_NAME}/input_data/cleaned_data.csv"
BQ_PROJECT = "iitj-capstone-project-group-18"
BQ_DATASET = "CAPSTONE_PROJECT"
BQ_TABLE = "HOTEL_BOOKING"
BQ_PREDICTION_TABLE = "PREDICTION_RESULTS"
BUCKET_NAME = "capstone-model-group-18"
CANCELLATION_MODEL_PATH = "Models/gradient_boosting_cancellation_model.pkl"
ADR_MODEL_PATH ="Models/gradient_boosting_regressor_adr_model.pkl"
ADR_CACHED_MODEL = None
CANCELLATION_CACHED_MODEL = None
SQL_QUERY = """SELECT guid,lead_time, hotel, market_segment, previous_cancellations,is_canceled,adr,is_repeated_guest,
            booking_changes, total_of_special_requests, arrival_date_month FROM `CAPSTONE_PROJECT.HOTEL_BOOKING`;"""

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
        {'name': 'is_repeated_guest', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name': 'is_canceled', 'type': 'BOOLEAN', "mode": "NULLABLE"},
        {'name': 'predicted_is_canceled', 'type': 'BOOLEAN', "mode": "NULLABLE"},
        {'name': 'adr', 'type': 'FLOAT', "mode": "NULLABLE"},
        {'name': 'predicted_adr', 'type': 'FLOAT', "mode": "NULLABLE"},
        {'name': 'insert_timestamp', 'type': 'TIMESTAMP', "mode": "NULLABLE"}
    ]
}

def load_data_from_gcs(file_path):
    """Load data from GCS into a pandas DataFrame."""
    logging.info("Loading Data from GCS to be ingested")
    return pd.read_csv(file_path)

def load_data_from_bigquery(query):
    """Load data from Bigquery Table into a pandas Dataframe"""
    try:
        logging.info("Connecting to big query")
        client = bigquery.Client(project=BQ_PROJECT)
        query_job = client.query(query)
        df = query_job.to_dataframe()
        return df
    except Exception as e:
        logging.error(f"Failed to load the data from Bigquery with the following error {str(e)}")
        
    
def load_model_from_gcs(bucket_name, model_path,task):
    """Load a model from GCS if not already cached."""
    logging.info(f"Loading model for task: {task}")
    if task == "adr":
        global ADR_CACHED_MODEL
        CACHED_MODEL = ADR_CACHED_MODEL
    elif task == "cancellation":
        global CANCELLATION_CACHED_MODEL
        CACHED_MODEL = CANCELLATION_CACHED_MODEL
    else:
        CACHED_MODEL = None
        
    if CACHED_MODEL is None:  # Check if the model is already cached
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(model_path)

        # Read the model file from GCS
        with blob.open("rb") as model_file:
            CACHED_MODEL = joblib.load(model_file)
        logging.info(f"Model loaded successfully for task: {task}")
    else:
        logging.info("Using cached model...")
    return CACHED_MODEL

def generate_synthetic_data(input_df):
    """Generate synthetic data with random sampling and add metadata."""
    # Randomly determine the number of rows to generate (between 0 and 100)
    logging.info("Generating Synthetic data to be inserted into Big Query")
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
    logging.info("Generation Complete ")
    return synthetic_data

def label_encoder(ip_data):
    label_encoders = {}
    for col in ['hotel', 'market_segment', 'arrival_date_month']:
        le = LabelEncoder()
        ip_data[col] = le.fit_transform(ip_data[col])
        label_encoders[col] = le
    return ip_data
        
def generate_predictions_with_model(df):
    """Generate predictions using a loaded model and add metadata."""
    input_df = df.drop('guid',axis=1)
    logging.info(f"Starting prediction process for Cancellation")
    cancellation_model = load_model_from_gcs(BUCKET_NAME, CANCELLATION_MODEL_PATH,"cancellation")
    # Features expected by the model
    cancellation_features = [
        "lead_time", "hotel", "market_segment", 
        "previous_cancellations", "booking_changes", 
        "total_of_special_requests", "arrival_date_month"
    ]
    prediction_df = input_df[cancellation_features + ['is_canceled']].dropna()
    label_encoder(prediction_df)
    prediction_df['is_canceled'] = prediction_df['is_canceled'].apply(lambda x: 1 if x == 'yes' else 0)
    try:
        df["predicted_is_canceled"] = cancellation_model.predict(prediction_df[cancellation_features])
        logging.info('Prediction of Cancellation ran sucessfully ')
    except Exception as e:
        logging.error(f"Failed to predict the error with the following error {str(e)}")
    # Load ADR model
    print("ADR task running")
    adr_model = load_model_from_gcs(BUCKET_NAME, ADR_MODEL_PATH,"adr")
    adr_features  = [
    'hotel', 'lead_time', 'market_segment', 
    'arrival_date_month', 'previous_cancellations', 
    'booking_changes', 'total_of_special_requests', 
    'is_repeated_guest']   
    prediction_df = input_df[adr_features + ['adr']].dropna()
    label_encoder(prediction_df)
    try:
        df["predicted_adr"] = adr_model.predict(prediction_df[adr_features])
        logging.info('Prediction of adr ran sucessfully ')
    except Exception as e:
        logging.error(f"Failed to predict the error with the following error {str(e)}")
    ind_zone = pytz.timezone("Asia/Kolkata")
    df["insert_timestamp"] = datetime.now(ind_zone)
    return df
    
def write_to_bigquery(df, project_id, dataset_id, table_id, schema):
    """Write the DataFrame to a BigQuery table."""
    logging.info("Creating a client to write into Big query")
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
    logging.info(f"Data successfully written into to {table_ref}")
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

@app.route('/',methods=['GET'])
def dummy():
    return jsonify({"status":"Success","message":"Now try using /ingest to ingest data or /predict to run prediction model"})

@app.route('/predict', methods=['POST'])
def preditct():
    """Trigger prediction pipeline for ADR."""
    try:
        # Step 1: Load data from Bigquery
        input_df =  load_data_from_bigquery(SQL_QUERY)
        # Step 2: Generate predictions
        prediction_df = generate_predictions_with_model(input_df)
        result = write_to_bigquery(prediction_df, BQ_PROJECT, BQ_DATASET, BQ_PREDICTION_TABLE, PREDICTION_BIGQUERY_SCHEMA)
        return jsonify({"status": "success", "message": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
