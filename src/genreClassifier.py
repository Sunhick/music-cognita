__author__ = "Sunil"
__copyright__ = "Copyright 2016"
__license__ = "MIT License"
__version__ = "0.1.0"
__email__ = "suba5417@colorado.edu"

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
from pyspark.mllib.classification import SVMModel

class GenreClassifier(object):
	""" Genre classifier """
	def __init__(self, file, appName, memSize, silent_mode):
		conf = ps.SparkConf().setAppName(appName).setMaster('local') \
					.set("spark.executor.memory", memSize)
		self.ctx = ps.SparkContext(conf=conf)
		
		if silent_mode:
			self.silentMode(self.ctx)

		self.sqlContext = SQLContext(self.ctx)

	def silentMode(self, ctx):
		print 'Turning on silent mode'
		logger = ctx._jvm.org.apache.log4j
		logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
		logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
		logger.LogManager.getLogger("INFO").setLevel(logger.Level.OFF)

	def parseData(line):
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