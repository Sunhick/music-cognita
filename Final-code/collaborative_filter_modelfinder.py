##################################################
# Desc  : colloborative based filtering
#         Find the best ALS Model and save it
##################################################

# !/usr/bin/python
import argparse
import itertools
import math
import triplet_parser as tp
import pyspark as ps
from pyspark.mllib.recommendation import ALS


class CollaborativeFilter(object):
    def __init__(self, dataFile, appName, memSize, checkpointPath, modelPath):
        self.dataFile = 'file:///{}'.format(dataFile)
        self.modelPath ='file:///{}'.format(modelPath)
        self.conf = ps.SparkConf().setAppName(appName).set("spark.executor.memory", memSize)
        self.sc = ps.SparkContext(conf=self.conf)
        if checkpointPath is not None:
            self.checkpointPath = 'file:///{}'.format(checkpointPath)
            self.sc.setCheckpointDir(checkpointPath)
        else:
            self.sc.setCheckpointDir(r'\check')

        #print dataFile, appName, memSize, checkpointPath, modelPath

    def trainModel(self):
        data_triplets = self.loadRatings(self.sc, self.dataFile)

        #Split the data into 80% training data and 20% valdiation data
        train_triplet = data_triplets.sample(False,0.8, seed=1).cache()
        validation_triplets = data_triplets.subtract(train_triplet).cache()

        print 20 * '-', 'Started Training the ALS model', 20 * '-'
        #TODO set different ranks and lambdas
        ranks = [8, 10, 12]
        lambdas = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]
        numIters = [20]
        alp = [1.0,10.0, 50.0, 100.0]
        bestModel = None
        bestValidationRMSE = float("inf")
        bestRank = 0
        bestLambda = -1.0
        bestNumIter = -1
        bestalpha = -1.0


        for rank, lmbda, numIter, a in itertools.product(ranks, lambdas, numIters, alp):

            print ("\nTraining ALS with rank = {}, Regularization parameter = {}, \n"
                   "number of iterations = {}, alpha = {}".format(rank, lmbda, numIter, a))

            model = ALS.trainImplicit(train_triplet, rank, lambda_=lmbda, iterations=numIter, alpha=a)

            #testdata contains only userId and songId
            testdata = validation_triplets.map(lambda r: (r[0], r[1]))

            #prediciton will contain userId, songId and predicted ratings
            predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))

            #Join the predicted ratings with actual rating to compute root mean square error
            actualsAndPredictions = validation_triplets.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
            RMSE = math.sqrt(actualsAndPredictions.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())
            print ('\n RMSE = {} ').format(RMSE)

            if (RMSE < bestValidationRMSE):
                bestModel = model
                bestValidationRMSE = RMSE
                bestRank = rank
                bestLambda = lmbda
                bestNumIter = numIter
                bestalpha = a

        print ("\nThe best model was trained with Rank = {}, Regularization parameter ={} and"
               "\nNumber of Iterations = {}\n RMSE = {}, best alpha = {}"
               .format(bestRank, bestLambda, bestNumIter, bestValidationRMSE, bestalpha))

        print 20 * '-', 'Finished Training the ALS model', 20 * '-'


        # Evaluate the best model on the test set. Use the entire data file as test set.


        print(20 * '-', 'Testing on the given data file itself', 20 * '-')

        test_ratings = self.loadRatings(self.sc, self.dataFile)
        testdata = test_ratings.map(lambda p: (p[0], p[1]))
        predictions = bestModel.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
        ratesAndPreds = test_ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
        MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
        MAE = ratesAndPreds.map(lambda r: (abs(abs(r[1][0]) - abs(r[1][1])))).mean()

        print("Mean Squared Error = " + str(MSE))
        print("Mean Absolute Error = " + str(MAE))
        print("Root Mean Square Error = ", str(MSE ** .5))
        print(20 * '-', 'Testing Finished', 20 * '-')

        # Save the best model

        #bestModel.save(self.sc, self.modelPath)
        bestModel.save(self.sc, self.modelPath)

    def loadRatings(self, sc, data_file):
        data = sc.textFile(data_file)
        triplet_in_dict_form, data_triplet = tp.format_triplets(data)

        play_count = data_triplet.count()
        unique_user_count = data_triplet.map(lambda r: r[0]).distinct().count()
        unique_songs_count = data_triplet.map(lambda r: r[1]).distinct().count()

        print("\nTotal play count of Songs: {}\nTotal number of Unique songs:{}\n"
              "Total number of distinct users {}".format(play_count, unique_songs_count, unique_user_count, ))
        print(100 * '-')

        return data_triplet


def main(args):
    kwargs = {
        'appName': 'Collaborative Filter',
        'memSize': '30g',
        'dataFile': args.datafile,
        'checkpointPath': args.checkpoint,
        'modelPath': args.modelpath
    }

    collabFilter = CollaborativeFilter(**kwargs)
    collabFilter.trainModel()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A Collaborative Filter to find the best ALS model for given music ratings')
    parser.add_argument('-d', '--datafile', help='Data File containing music ratings in for of\n'
                                                 'userId\SongId\tPlayCounts', required=True)
    parser.add_argument('-c', '--checkpoint', help='Checkpoint Path to save RDDs', required=False)
    parser.add_argument('-m', '--modelpath', help='Path to Save the ALS model', required=True)
    arguments = parser.parse_args()

    args = parser.parse_args()
    main(args)
