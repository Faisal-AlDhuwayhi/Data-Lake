import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import TimestampType

# load AWS credentials as environment variables
config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """
        Create and set the configuration of a spark session.
        
        Returns:
        1) spark (object): a spark session object
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
        Read song data from data lake and process it into songs and artists tables,
        then load it back to the S3 data lake buckets as parquet files.
        
        Args:
        1) spark (object): a spark session object
        2) input_data (str): the S3 input data bucket path
        3) output_data (str): the S3 output data bucket path
    """
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select(['song_id', 'title', 'artist_id', 'year', 'duration']).dropDuplicates(["song_id"])
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.parquet(output_data + 'songs_table.parquet', partitionBy=['year', 'artist_id'])

    # extract columns to create artists table
    artists_table = df.select(["artist_id", "artist_name", "artist_location", "artist_latitude", "artist_longitude"]).dropDuplicates(["artist_id"])
    
    # write artists table to parquet files
    artists_table.write.parquet(output_data + 'artists_table.parquet')


def process_log_data(spark, input_data, output_data):
    """
        Read log data from data lake and process it into users, time, and songplays tables,
        then load it back to the S3 data lake buckets as parquet files.
        
        Args:
        1) spark (pyspark session object): a spark session object
        2) input_data (str): the S3 input data bucket path
        3) output_data (str): the S3 output data bucket path
    """
    # get filepath to log data file
    log_data = os.path.join(input_data, 'log_data/*/*/*.json')

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter(col('page') == 'NextSong')

    # extract columns for users table    
    users_table = df.selectExpr(["userId as user_id", "firstName as first_name", "lastName as last_name", "gender", "level"]).dropDuplicates(["user_id"]) 
    
    # write users table to parquet files
    users_table.write.parquet(output_data + "users_table.parquet")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp((x / 1000)), TimestampType())
    df = df.withColumn("timestamp", get_timestamp(col("ts")))
    
    # extract columns to create time table
    time_table = df.selectExpr("timestamp as start_time",
                               "hour(timestamp) as hour",
                               "dayofmonth(timestamp) as day",
                               "weekofyear(timestamp) as week",
                               "month(timestamp) as month",
                               "year(timestamp) as year",
                               "dayofweek(timestamp) as weekday"
                               ).dropDuplicates(["start_time"])
    
    # write time table to parquet files partitioned by year and month
    time_table.write.parquet(output_data + "time_table.parquet", partitionBy=['year', 'month'])

    # read in song data to use for songplays table
    song_df = spark.read.json(os.path.join(input_data, 'song_data/*/*/*/*.json'))

    ## extract columns from joined song and log datasets to create songplays table 
    # create a view to use it for in the SQL join expression
    song_df.createOrReplaceTempView("songs")
    df.createOrReplaceTempView("events")

    songplays_table = spark.sql("""
                                SELECT 
                                    monotonically_increasing_id() as songplay_id,
                                    e.timestamp as start_time,
                                    year(e.timestamp) as year,
                                    month(e.timestamp) as month,
                                    e.userId as user_id,
                                    e.level,
                                    s.song_id,
                                    s.artist_id,
                                    e.sessionId as session_id,
                                    e.location,
                                    e.userAgent as user_agent

                                FROM events e
                                JOIN songs s
                                ON e.song = s.title AND e.artist = s.artist_name AND ABS(e.length - s.duration) < 2
                                """)

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.parquet(output_data + 'songplays_table.parquet', partitionBy=['year', 'month'])


def main():
    # create the spark session and specify the input and output data paths
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://dend-data-lake-bucket/"
    
    # run the ETL pipeline on the song and log data 
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)

if __name__ == "__main__":
    main()
