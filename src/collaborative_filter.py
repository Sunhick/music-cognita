##################################################
# Author: sunil bn
# Email : suba5417@colorado.edu
#
# Desc  : colloborative based filtering 
#         technique for music recommender
##################################################

#!/usr/bin/python

import os
import argparse

import pyspark as ps
import numpy as np


class CollaborativeFilter(object):
	def __init__(self, train_file, appName, memSize, silent_mode):
		self.trainFile = train_file
		conf = ps.SparkConf().setAppName(appName) \
					.set("spark.executor.memory", memSize)
		self.ctx = ps.SparkContext(conf=conf)

		if silent_mode:
			self.silent_mode(self.ctx)

	def silent_mode(self, ctx):
		print 'Turning on silent mode'
		logger = ctx._jvm.org.apache.log4j
		logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
		logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
		logger.LogManager.getLogger("INFO").setLevel(logger.Level.OFF)

	def trainModel(self):
		pass

	def saveModel(self):
		pass

def main(args):
	kwargs = {
		'appName': 'Collaborative Filter', 
		'memSize': '5g', 
		'silent_mode':False if args.verbose > 0 else True, 
		'train_file': args.train
		}

	collabFilter = CollaborativeFilter(**kwargs)
	collabFilter.trainModel()
	collabFilter.saveModel()


if __name__ == '__main__':
	parser  = argparse.ArgumentParser(description='Collaborative filtering for music recommender')
	parser.add_argument('--train', type=str, help='python collaborative_filter <path_to_dataset>')
	parser.add_argument('--test',  type=str, help='python collaborative_filter <path_to_dataset>')
	parser.add_argument('-v', '--verbose', action='count', help='verbosity')

	args = parser.parse_args()
	main(args)