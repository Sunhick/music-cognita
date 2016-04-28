import argparse
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert Triplets data - txt to Parquet')
    parser.add_argument('-d', '--datapath', help='Folder where Triple file is stored in csv format', required=True)
    args = parser.parse_args()
    database = args.datapath

    conf = SparkConf().setAppName("Convert to Parquet").set("spark.executor.memory", '3g')
    sc = SparkContext(conf=conf)
    sq = SQLContext(sc)

    lines = sc.textFile(database + r"\train_triplets.txt")

    words = lines.map(lambda l: l.split("\t"))
    ratings = words.map(lambda w: (w[0], w[1], int(w[2])))

    schema = StructType([StructField("user_id", StringType(), False),
                         StructField("song_id", StringType(), False),
                         StructField("play_count", IntegerType(), True)])

    # Apply the schema to the RDD.
    df_schemaRatings = sq.createDataFrame(ratings, schema).cache()

    df_schemaRatings.write.parquet(database + r"\triplets_data")
    sc.stop()