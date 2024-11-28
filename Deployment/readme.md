
# Hotel Booking Prediction Pipeline

## Overview

This project provides a pipeline for ingesting hotel booking data, generating synthetic data, running predictions using machine learning models, and storing results in Google BigQuery. It is designed for deployment on Google Cloud Platform (GCP).

### Features
1. **Data Ingestion**:
   - Reads hotel booking data from Google Cloud Storage (GCS).
   - Generates synthetic data based on real data for testing or analysis.
   - Stores ingested data in BigQuery.

2. **Predictions**:
   - Uses two ML models for predictions:
     - **Cancellation Prediction Model**: Predicts whether a booking will be canceled.
     - **Average Daily Rate (ADR) Prediction Model**: Predicts the ADR for hotel bookings.
   - Results are appended to a BigQuery table.

3. **Logging**:
   - Logs pipeline events to a file and console for debugging and monitoring.

4. **API Endpoints**:
   - `/ingest`: Initiates the data ingestion pipeline.
   - `/predict`: Runs the prediction pipeline.
   - `/`: Health check endpoint.

---

## File Descriptions

### 1. `main.py`
This is the core Flask application for the pipeline. Key functionalities include:
- **Data Handling**:
  - Reads data from GCS and BigQuery.
  - Generates synthetic data for testing.
- **Model Predictions**:
  - Loads ML models from GCS.
  - Generates predictions for cancellation and ADR.
- **BigQuery Operations**:
  - Writes ingested and predicted data to respective BigQuery tables.
- **Endpoints**:
  - `/ingest`: Triggers the ingestion pipeline.
  - `/predict`: Runs predictions and stores results.

### 2. `requirements.txt`
Contains the Python dependencies for the project, including:
- `Flask`: For API development.
- `pandas`, `numpy`: For data processing.
- `scikit-learn`: To load and use pre-trained ML models.
- `google-cloud-*`: For GCP operations like BigQuery and GCS interactions.
- `gunicorn`: To deploy the application in production.

### 3. `Dockerfile`
Builds a Docker container for the application:
- Installs dependencies from `requirements.txt`.
- Sets up the Flask app to run on port 8080 using Gunicorn.

### 4. `gcp_deployment.yaml`
Configures deployment to Google Cloud Run:
- Specifies the container image and service details.
- Integrates the app with GCP resources like BigQuery and GCS.

---

## Setup Instructions

### 1. Prerequisites
- **Google Cloud Project** with the following services enabled:
  - BigQuery
  - Cloud Storage
  - Cloud Run
- **GCP Credentials**:
  - Set up a service account with permissions for BigQuery and GCS.
  - Download the service account key JSON file and set the environment variable:
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
    ```

### 2. Local Development
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python main.py
   ```
   Access the application at `http://localhost:8080`.

### 3. Containerization
1. **Build Docker Image**:
   ```bash
   docker build -t hotel-booking-pipeline .
   ```

2. **Run Docker Container**:
   ```bash
   docker run -p 8080:8080 hotel-booking-pipeline
   ```

### 4. Deployment to GCP
1. **Build and Push Container to GCP Artifact Registry**:
   ```bash
   gcloud builds submit --tag gcr.io/<your-project-id>/hotel-booking-pipeline
   ```

2. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy hotel-booking-pipeline \
       --image gcr.io/<your-project-id>/hotel-booking-pipeline \
       --platform managed \
       --region <your-region> \
       --allow-unauthenticated
   ```

---

## API Usage

### 1. Health Check
**Endpoint**: `/`  
**Method**: `GET`  
**Response**:
```json
{
  "status": "Success",
  "message": "Now try using /ingest to ingest data or /predict to run prediction model"
}
```

### 2. Data Ingestion
**Endpoint**: `/ingest`  
**Method**: `POST`  
**Response**:
- **Success**:
  ```json
  {
    "status": "success",
    "message": "Data successfully written to <BigQuery table>"
  }
  ```
- **Error**:
  ```json
  {
    "status": "error",
    "message": "<error details>"
  }
  ```

### 3. Predictions
**Endpoint**: `/predict`  
**Method**: `POST`  
**Response**:
- **Success**:
  ```json
  {
    "status": "success",
    "message": "Data successfully written to <BigQuery prediction table>"
  }
  ```
- **Error**:
  ```json
  {
    "status": "error",
    "message": "<error details>"
  }
  ```

---

## Notes
- Ensure that the `BUCKET_NAME`, `BQ_PROJECT`, and related constants in `main.py` match your GCP configuration.
- The ML models must be uploaded to the specified GCS bucket paths.
- Update `gcp_deployment.yaml` with the correct GCP project details before deployment.

--- 

#### Contributer 
- Jeyadev L