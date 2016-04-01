##################################################
# Author: sunil bn
# Email : suba5417@colorado.edu
#
# Desc  : colloborative based filtering 
#         technique for music recommender
##################################################

#!/usr/bin/python

import os
import shutil
import argparse
import itertools

import pyspark as ps
import numpy as np

from src.parser import parse_line
from src.normalize import by_max_count, format_triplets
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

class CollaborativeFilter(object):
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

	def trainModel(self):
		train_ratings = self._loadRatings(self.ctx, self.trainFile)
		ratings_valid = train_ratings.sample(False, 0.1, 12345)
		ratings_train = train_ratings.subtract(ratings_valid)

		print(20*'-','TRAINING STARTED',20*'-')
		ranks = [8]
		lambdas = [1.0, 10.0, 5.0]
		numIters = [10]
		bestModel = None
		bestValidationMSE = float("inf")
		bestRank = 0
		bestLambda = -1.0
		bestNumIter = -1

		for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
			print(rank, lmbda, numIter)
			model = ALS.train(ratings_train, rank, numIter, lmbda)
			testdata = ratings_valid.map(lambda p: (p[0], p[1]))
			predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
			ratesAndPreds = ratings_valid.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
			MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()

			if (MSE < bestValidationMSE):
				bestModel = model
				bestValidationMSE = MSE
				bestRank = rank
				bestLambda = lmbda
				bestNumIter = numIter

		# evaluate the best model on the test set
		#model = ALS.train(ratings, rank, numIterations)
		print(20*'-','TRAINING FINISHED',20*'-')

		# # Evaluate the model on testing data
		print(20*'-','TESTING STARTED',20*'-')
		#TODO: stop gap for evaluation. using trainFile itself as test file
		test_ratings = self._loadRatings(self.ctx, self.trainFile)
		testdata = test_ratings.map(lambda p: (p[0], p[1]))
		predictions = bestModel.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
		ratesAndPreds = test_ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
		MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
		MAE = ratesAndPreds.map(lambda r: (abs(abs(r[1][0]) - abs(r[1][1])))).mean()

		print("Mean Squared Error = " + str(MSE))
		print("Mean Absolute Error = " + str(MAE))
		print("Root Mean Square Error = ", str(MSE**.5))
		print(20*'-','TESTING FINISHED',20*'-')


	def saveModel(self):
		pass

	def _loadRatings(self, sc, data_file):
		data = sc.textFile(data_file)
		data_dict, data_triplet = format_triplets(data)
		data_triplet = by_max_count(data_triplet)
		num_ratings = data_triplet.count()
		num_users = data_triplet.map(lambda r: r[0]).distinct().count()
		num_songs = data_triplet.map(lambda r: r[1]).distinct().count()
		print(100 * '//')
		print("Got {} ratings, with {} distinct songs and {} distinct users".format(num_ratings,
        	                                                                        num_users,
            	                                                                    num_songs))
		print(100 * '//')
		train_ratings = data_triplet.map(lambda l: Rating(l[0], l[1], l[2]))
		return train_ratings

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