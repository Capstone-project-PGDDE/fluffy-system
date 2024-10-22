from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, FloatType, DateType

# Initializing PySpark session
spark = SparkSession.builder \
    .appName("KafkaSparkBigQuery") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.1") \
    .getOrCreate()

# Defining the full schema based on the BigQuery table DDL
schema = StructType([
    StructField("hotel", StringType(), True),
    StructField("is_canceled", BooleanType(), True),
    StructField("lead_time", IntegerType(), True),
    StructField("arrival_date_year", IntegerType(), True),
    StructField("arrival_date_month", StringType(), True),
    StructField("arrival_date_week_number", IntegerType(), True),
    StructField("arrival_date_day_of_month", IntegerType(), True),
    StructField("stays_in_weekend_nights", IntegerType(), True),
    StructField("stays_in_week_nights", IntegerType(), True),
    StructField("adults", IntegerType(), True),
    StructField("children", IntegerType(), True),
    StructField("babies", IntegerType(), True),
    StructField("meal", StringType(), True),
    StructField("country", StringType(), True),
    StructField("market_segment", StringType(), True),
    StructField("distribution_channel", StringType(), True),
    StructField("is_repeated_guest", BooleanType(), True),
    StructField("previous_cancellations", IntegerType(), True),
    StructField("previous_bookings_not_canceled", IntegerType(), True),
    StructField("reserved_room_type", StringType(), True),
    StructField("assigned_room_type", StringType(), True),
    StructField("booking_changes", IntegerType(), True),
    StructField("deposit_type", StringType(), True),
    StructField("agent", StringType(), True),
    StructField("company", StringType(), True),
    StructField("days_in_waiting_list", IntegerType(), True),
    StructField("customer_type", StringType(), True),
    StructField("adr", FloatType(), True),
    StructField("required_car_parking_spaces", IntegerType(), True),
    StructField("total_of_special_requests", IntegerType(), True),
    StructField("reservation_status", StringType(), True),
    StructField("reservation_status_date", DateType(), True)
])

df = spark \
  .readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", "localhost:9092") \
  .option("subscribe", "hotel-bookings") \
  .load()

df = df.selectExpr("CAST(value AS STRING)")

df_parsed = df.withColumn("value", from_json(col("value"), schema)).select(col("value.*"))

# Print the full schema to ensure the whole table is being processed
df_parsed.printSchema()

# Optional: Write the entire parsed data stream to BigQuery or display in the console
# Writing to BigQuery
df_parsed.writeStream \
    .format("bigquery") \
    .option("table", "`capstone-project-group-18.Capstone_Source.hotel_bookings_stream`") \
    .option("checkpointLocation", "/tmp/checkpoints") \
    .start()

spark.streams.awaitAnyTermination()
