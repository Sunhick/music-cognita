import argparse
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create a new table for Sing ID and Hash Id reference')
    parser.add_argument('-d', '--datapath', help='Folder where Songs data is stored in Parquet format', required=True)
    args = parser.parse_args()
    database = args.datapath

    conf = SparkConf().setAppName("practice").set("spark.executor.memory", '3g')
    sc = SparkContext(conf=conf)
    sq = SQLContext(sc)

    df = sq.read.parquet(database + r"\songs_data").cache()

    temp = df.select(df.song_id).flatMap(list).map(lambda l: (l, int(hash(l) & 0xfffffff)))

    data = sc.parallelize(temp.collect())
    #print data.collect()

    schema = StructType([StructField("song_id", StringType(), True),
                         StructField("hash_id", LongType(), True)])

    df1 = sq.createDataFrame(data, schema).cache()
    print df1.count()
    df1.write.parquet(database + r"\song_hash_id")

    sc.stop()