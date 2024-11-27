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
COPY ingestion_pipeline.py /pipelines/
COPY Pred_canellation.py /pipelines/

# Set entrypoint
ENTRYPOINT ["python"]
