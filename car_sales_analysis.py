from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("CarSalesAnalysis") \
    .getOrCreate()

# Read CSV file with header
df = spark.read.csv("c:/Users/DELL/Desktop/Book2.csv", header=True)

# Show original schema and data
print("Original Schema:")
df.printSchema()
print("\nOriginal Data:")
df.show(truncate=False)

# Identify actual columns from header
actual_columns = [
    "Dealer_ID", 
    "Model_ID", 
    "Branch_ID", 
    "Date_ID", 
    "Units_Sold", 
    "Revenue"
]

# Select only actual columns and drop empty columns
cleaned_df = df.select([col(c) for c in actual_columns])

# Convert numeric columns to proper types
cleaned_df = cleaned_df.withColumn("Units_Sold", col("Units_Sold").cast("integer")) \
                       .withColumn("Revenue", col("Revenue").cast("integer"))

# Show cleaned schema and data
print("\nCleaned Schema:")
cleaned_df.printSchema()
print("\nCleaned Data:")
cleaned_df.show(truncate=False)

# Calculate summary statistics by model
summary_df = cleaned_df.groupBy("Model_ID") \
    .agg(
        sum("Units_Sold").alias("Total_Units_Sold"),
        sum("Revenue").alias("Total_Revenue")
    ) \
    .orderBy("Total_Revenue", ascending=False)

print("\nSales Summary by Model:")
summary_df.show(truncate=False)

# Stop Spark session
spark.stop()
