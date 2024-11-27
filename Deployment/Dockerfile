# Use the Apache Beam SDK base image
FROM apache/beam_python3.10_sdk:2.48.0

# Install required dependencies
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    google-cloud-bigquery \
    google-cloud-storage \
    pytz \
    uuid

# Copy the pipeline code into the container
WORKDIR /pipelines
COPY Deployment/ingestion_pipeline.py /pipelines/
COPY Deployment/Pred_Cancellation.py /pipelines/

# Set entrypoint
ENTRYPOINT ["python"]
