import argparse
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert Songs data - CSV to Parquet')
    parser.add_argument('-d', '--datapath', help='Folder where Songs data and attributes are stored in csv format', required=True)
    args = parser.parse_args()
    database = args.datapath

    conf = SparkConf().setAppName("Convert to Parquet").set("spark.executor.memory", '1g')
    sc = SparkContext(conf=conf)
    sq = SQLContext(sc)

    songs_meta_data = sc.textFile(database + r"\Songdata.csv")
    songs_attributes = sc.textFile(database + r"\song_attributes.csv")
    header = songs_meta_data.first()
    header2 = songs_attributes.first()
    print header
    print header2


    head = songs_meta_data.filter(lambda l: "track_id" in l)
    head2 = songs_attributes.filter(lambda l: "track_id" in l)
    #print head.collect()
    data = songs_meta_data.subtract(head)
    data2 = songs_attributes.subtract(head2)

    #print header
    fields = [StructField(field_name, StringType(), True) for field_name in header.split(',')]
    fields[7].dataType = FloatType() #duration
    fields[8].dataType = FloatType() #artist_familiarity
    fields[9].dataType = FloatType() #artist_hotttnesss
    fields[10].dataType = IntegerType() #year
    fields[11].dataType = IntegerType() #track_7digitalid
    fields[12].dataType = IntegerType() #shs_perf
    fields[13].dataType = IntegerType() #shs_work

    #print len(fields), '\n'

    schema = StructType(fields)
    #shema fields are ---> track_id,title,song_id,release,artist_id,artist_mbid,artist_name,duration,
    # artist_familiarity,artist_hotttnesss,year,track_7digitalid,shs_perf,shs_work

    data_temp = data.map(lambda k: k.split(","))\
        .map(lambda p: (p[0], p[1], p[2], p[3], p[4],
                        p[5], p[6], float(p[7]), float(p[8]), float(p[9]),
                        int(p[10]), int(p[11]), int(p[12]), int(p[13])))

    #print data_temp.collect()
    df_songs = sq.createDataFrame(data_temp, schema)

    #song attributes
    fields2 = [StructField(field_name, StringType(), True) for field_name in header2.split(',')]
    fields2[1].dataType = FloatType() #track_id
    fields2[2].dataType = FloatType() #danceability
    fields2[3].dataType = IntegerType() #key
    fields2[4].dataType = FloatType() #energy
    fields2[5].dataType = FloatType() #tempo
    fields2[6].dataType = IntegerType() #time_signature

    schema2 = StructType(fields2)
    #shema fields are ---> track_id,danceability,energy,key,loudness,tempo,time_signature

    data_temp2 = data2.map(lambda k: k.split(","))\
        .map(lambda p: (p[0], float(p[1]), float(p[2]), int(p[3]), float(p[4]), float(p[5]), int(p[6])))

    df_songs_attributes = sq.createDataFrame(data_temp2, schema2)
    combined = df_songs.join(df_songs_attributes, df_songs.track_id == df_songs_attributes.track_id).drop(df_songs_attributes.track_id)
    combined.show()
    combined.write.parquet(database + r"\songs_data")

    sc.stop()
