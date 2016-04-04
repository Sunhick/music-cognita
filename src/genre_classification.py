##################################################
# Author: sunil bn
# Email : suba5417@colorado.edu
#
# Desc  : Genre classification
##################################################

#!/usr/bin/python

__author__ = "Sunil"
__copyright__ = "Copyright 2016"
__license__ = "MIT License"
__version__ = "0.1.0"
__email__ = "suba5417@colorado.edu"

import argparse
import pyspark as ps
import numpy as np
import sys

from pyspark.sql import SQLContext
from pyspark.ml.feature import IndexToString
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import StringIndexer 
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

"""
# MILLION SONG GENRE DATASET
#
# http://labrosa.ee.columbia.edu
# GOAL: easy to use genre-like dataset extracted from the MSD
#       Should not be used as a proper MIR task! Educational purposes only
# FORMAT: # - denotes a comment
          % - one line after comments, column names
            - rest is data, comma-separated, one line per song
genre,track_id,artist_name,title,loudness,tempo,time_signature,key,mode,duration,avg_timbre1,avg_timbre2,avg_timbre3,avg_timbre4,avg_timbre5,avg_timbre6,avg_timbre7,avg_timbre8,avg_timbre9,avg_timbre10,avg_timbre11,avg_timbre12,var_timbre1,var_timbre2,var_timbre3,var_timbre4,var_timbre5,var_timbre6,var_timbre7,var_timbre8,var_timbre9,var_timbre10,var_timbre11,var_timbre12
  0    1         2			3		4		5		6			7	8		9		10			11			12			13			14			15			16			17			18			19				20			21			22			23			24			25			26			27			28			29			30			31			32			33
"""

def parse(line):
	line = line.split(',')
	genre = line[0]
	tid = line[1]
	loudness = float(line[4])
	tempo = float(line[5])
	timesig = float(line[6])
	key = float(line[7])
	mode = float(line[8])
	duration = float(line[9])

	avgtimbre = [float(line[i]) for i in range(10, 22)]
	vartimbre = [float(line[i]) for i in range(22, 34)]
	
	features = [loudness, tempo, timesig, key, mode, duration]
	features.extend(avgtimbre)
	features.extend(vartimbre)
	# t1 = float
	#return {'genre': genre, 'tid': tid, 'loudness': loudness, 'tempo': tempo, 'time_signature': timesig,
	#		'key': key, 'mode': mode, 'duration': duration}
	#return {'hash': int(hash(tid) & 0xfffffff), 'genre': genre, 'tid': tid, 'features': np.array([loudness, tempo, timesig, key, mode, duration])}
	return [genre, tid, Vectors.dense(features)]

def silentMode(ctx):
		print 'Turning on silent mode'
		logger = ctx._jvm.org.apache.log4j
		logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
		logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
		logger.LogManager.getLogger("INFO").setLevel(logger.Level.OFF)

def main(args):
	conf = ps.SparkConf().setAppName('genre_classification').setMaster('local')
	ctx = ps.SparkContext(conf=conf)

	silentMode(ctx)
	sqlContext = SQLContext(ctx)

	tf = ctx.textFile('/Users/Sunny/prv/github/music-cognita/data/MillionSongSubset/AdditionalFiles/msd_genre_dataset.txt')
	rdd = tf.map(parse)

	# The schema is encoded in a string.
	schema = ['genre', 'track_id', 'features']
	# Apply the schema to the RDD.
	songDF = sqlContext.createDataFrame(rdd, schema)

	# Register the DataFrame as a table.
	songDF.registerTempTable("genclass")
	labelIndexer = StringIndexer().setInputCol("genre").setOutputCol("indexedLabel").fit(songDF)

	trainingData, testData = songDF.randomSplit([0.8, 0.2])

	labelConverter = IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

	rfc = RandomForestClassifier().setMaxDepth(10).setNumTrees(3).setLabelCol("indexedLabel").setFeaturesCol("features")

	pipeline = Pipeline(stages=[labelIndexer, rfc, labelConverter])
	model = pipeline.fit(trainingData)

	predictions = model.transform(testData)
	predictions.show()

	evaluator = MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("precision")
	accuracy = evaluator.evaluate(predictions)
	print "Test Error = ", (1.0 - accuracy)

if __name__ == '__main__':
	parser  = argparse.ArgumentParser(description='Genre classification')
	parser.add_argument('--file', type=str, help='python genre_classification <path_to_dataset>')
	parser.add_argument('-v', '--verbose', action='count', help='verbosity')

	args = parser.parse_args()
	main(args)