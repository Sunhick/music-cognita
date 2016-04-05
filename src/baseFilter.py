__author__ = "Sunil"
__copyright__ = "Copyright 2016"
__license__ = "MIT License"
__version__ = "0.1.0"
__email__ = "suba5417@colorado.edu"


class Filter(object):
	""" base filter for all types of filtering techniques """
	def __init__(self, train_file, appName, memSize, silent_mode):
		self.trainFile = 'file:///{}'.format(train_file)
		conf = ps.SparkConf().setAppName(appName) \
					.set("spark.executor.memory", memSize)
		self.ctx = ps.SparkContext(conf=conf)

		if silent_mode:
			self.silentMode(self.ctx)

	def silentMode(self, ctx):
		print 'Turning on silent mode'
		logger = ctx._jvm.org.apache.log4j
		logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
		logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
		logger.LogManager.getLogger("INFO").setLevel(logger.Level.OFF)
