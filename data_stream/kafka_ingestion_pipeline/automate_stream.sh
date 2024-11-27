#!/bin/bash

# Define the paths to Kafka and your scripts
KAFKA_DIR="D:\kafka_2.12-3.8.0"
DATA_STREAM_DIR="C:\Users\GOD\Desktop\Python Program\fluffy-system\data_stream"
# Start Zookeeper
start_zookeeper() {
  echo "Starting Zookeeper..."
  "$KAFKA_DIR/bin/windows/zookeeper-server-start.bat" "$KAFKA_DIR/config/zookeeper.properties" &
  ZOOKEEPER_PID=$!
  sleep 10  # Give Zookeeper some time to start
}

# Start Kafka Broker
start_kafka_broker() {
  echo "Starting Kafka Broker..."
  # Remove any lingering lock files to avoid startup issues
  if [ -f "$KAFKA_DIR/kafka-logs/.lock" ]; then
    echo "Removing existing Kafka lock file..."
    rm "$KAFKA_DIR/kafka-logs/.lock"
  fi
  "$KAFKA_DIR/bin/windows/kafka-server-start.bat" "$KAFKA_DIR/config/server.properties" &
  KAFKA_PID=$!
  sleep 20  # Give Kafka broker more time to start
}

# Check Kafka Broker Availability
check_kafka_broker() {
  echo "Checking if Kafka Broker is available..."
  MAX_RETRIES=5
  RETRY_COUNT=0
  while ! (echo > /dev/tcp/localhost/9092) 2>/dev/null && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "Waiting for Kafka Broker to be ready..."
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT + 1))
  done

  if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Kafka Broker is not available. Exiting."
    stop_kafka_services
    exit 1
  fi
  echo "Kafka Broker is available."
}

# Run Data Sender Python Script
run_data_sender() {
  echo "Running Data Sender..."
  python "$DATA_STREAM_DIR/data_sender.py" &
  DATA_SENDER_PID=$!
}

# Run Pandas Stream Python Script
run_pandas_stream() {
  echo "Running Pandas Stream..."
  python "$DATA_STREAM_DIR/pandas_stream.py" &
  PANDAS_STREAM_PID=$!
}

# Stop Zookeeper and Kafka Broker
stop_kafka_services() {
  echo "Stopping Kafka Broker..."
  if [[ -n "$KAFKA_PID" ]]; then
    kill -9 "$KAFKA_PID"
    echo "Kafka Broker stopped."
  fi

  echo "Stopping Zookeeper..."
  if [[ -n "$ZOOKEEPER_PID" ]]; then
    kill -9 "$ZOOKEEPER_PID"
    echo "Zookeeper stopped."
  fi
}

# Handle Ctrl+C to stop services
trap stop_kafka_services SIGINT

# Menu to choose an option
echo "Please choose an option:"
echo "1. Start Zookeeper, Kafka Broker, and run Python scripts"
echo "2. Stop Kafka and Zookeeper"
read -p "Enter option: " OPTION

case $OPTION in
  1)
    start_zookeeper
    start_kafka_broker
    check_kafka_broker
    run_data_sender
    run_pandas_stream
    wait  # Wait to keep the script running until interrupted
    ;;
  2)
    stop_kafka_services
    ;;
  *)
    echo "Invalid option. Exiting."
    ;;
esac
