import argparse
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert Songs Similarity data - CSV to Parquet')
    parser.add_argument('-d', '--datapath', help='Folder where Songs Similarity Data are stored in csv format', required=True)
    args = parser.parse_args()
    database = args.datapath

    conf = SparkConf().setAppName("Convert to Parquet").set("spark.executor.memory", '3g')
    sc = SparkContext(conf=conf)
    sq = SQLContext(sc)

    file = sc.textFile(database + r"\lastfm_similars_dest.csv").persist()

    header = file.first()
    head = file.filter(lambda l: "tid" in l)

    data = file.subtract(head)

    schema = StructType([StructField("tid", StringType(), True),
                         StructField("target", StringType(), True)])

    data_temp = data.map(lambda k: k.split(",",1))\
        .map(lambda p: (p[0], p[1].strip('"')))

    # Apply the schema to the RDD.
    df_similar = sq.createDataFrame(data_temp, schema).cache()
    #df_similar.show(truncate=False)

    df_similar.write.parquet(database + r"\song_similarity")
    sc.stop()