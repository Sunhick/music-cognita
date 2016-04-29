import argparse
from collections import defaultdict
from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.recommendation import MatrixFactorizationModel, Rating
from pyspark.sql import SQLContext, Row
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint


def logistic_train(line):
    dance = float(line[1])*10
    energy = float(line[2])*10
    key = float(line[3])
    loudness = float(line[4])
    tempo = float(line[5])
    time_sig = int(line[6])
    rating_class = round(int(line[7]))
    features = [dance, energy, key, loudness, tempo, time_sig]

    return LabeledPoint(rating_class, Vectors.dense(features))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A Content Based Filter to recommend songs')
    parser.add_argument('-d', '--datapath', help='Path where data is stored Apache Parquet format', required=True)
    parser.add_argument('-m', '--modelpath', help='Path to where the ALS model was saved', required=True)
    args = parser.parse_args()
    database = args.datapath

    conf = SparkConf().setAppName("CB filter").set("spark.executor.memory", '3g')
    sc = SparkContext(conf=conf)
    sq = SQLContext(sc)

    # load songs data
    df_songs = sq.read.parquet(database + r"\songs_data2")

    # load user play data i.e triplets
    df_triplets = sq.read.parquet(database + r"\triplets_data2")

    # load song_id and its hash_id
    df_song_hash_id = sq.read.parquet(database + r"\song_hash_id")

    # load song similarity data
    df_song_sim = sq.read.parquet(database + r"\song_similarity2")

    # load the ALS model
    ALSmodel = MatrixFactorizationModel.load(sc, args.modelpath)

    # Take an input from user
    # userid = '3cd99bb95d2baac1e910a5c847e58388d5e9b3c1'
    # userid = 'ef484f5d1c2bfe2eac0098ae460b793833b5acbc'

    userid = raw_input("\n Enter an UserID: ")

    # Convert the userid into its hash value
    user_hash = int(hash(userid) & 0xfffffff)

    #### display user statistcs
    # Find the songs listened by the user

    played_songs = df_triplets.filter(df_triplets.user_id == userid) \
        .select(df_triplets.song_id, df_triplets.play_count).cache()

    if played_songs.count() == 0:
        print "\n User Id not found"
        exit()
    # played_songs.sort(df_triplets.song_id.desc()).show(1000)


    # Run ALS Algorithm
    user_songs_id = played_songs.select(df_triplets.song_id).cache()

    # Convert Song id to hash id for ALS model

    step = 10
    for i in [10, 50, 100, 200, 500, 1000, 10000]:

        print "\n Songs to recommend = ", i
        ALS_predictions = ALSmodel.recommendProducts(user_hash, i)
        # print ALS_predictions

        max_ALS_rating = ALS_predictions[0].rating

        # Convert to RDD and normalize rating from 0 to 10
        df_ALS_predictions = sc.parallelize(ALS_predictions) \
            .map(lambda l: Rating(l.user, l.product, l.rating / max_ALS_rating * 10)) \
            .toDF().cache()

        # df_ALS_predictions.show()

        # Convert df_ALS_predictions which has hash values into id values
        df_ALS_songs = df_ALS_predictions.join(df_song_hash_id, df_ALS_predictions.product == df_song_hash_id.hash_id) \
            .select(df_song_hash_id.song_id, df_ALS_predictions.rating).dropDuplicates().cache()

        df_ALS_songs = df_ALS_songs.join(df_songs, df_songs.song_id == df_ALS_songs.song_id)\
            .dropDuplicates(['title', 'artist_name']).drop(df_ALS_songs.song_id).cache()


        df_ALS_songs = df_ALS_songs.withColumnRenamed('rating', 'predicted_score').cache()


        # Predict recommendation using Logistic regression. Pass ALS output here

        df_ALS_attributes = df_ALS_songs.select(df_ALS_songs.track_id, "danceability",
                            "energy", "key", "loudness", "tempo", "time_signature", "predicted_score")\
                            .where(df_ALS_songs.predicted_score > 0).cache()

        train_data_from_ALS = df_ALS_attributes.sample(False, 0.8, seed=1).cache()
        test = df_ALS_attributes.subtract(train_data_from_ALS).cache()

        train_data_from_ALS = df_ALS_attributes.map(lambda l: (l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7])).sortBy(lambda x:x[7]).map(
            logistic_train).cache()

        #print train_data_from_ALS.take(1000)

        print '\n', 20 * '-', "Start Training LogisticRegressionWithLBFGS using ALS predicted songs", 20 * '-'

        #number of class = 11 because ratings can be 0,1,2,3,4....,9,10
        LR_model = LogisticRegressionWithLBFGS.train(train_data_from_ALS, numClasses=11)

        print '\n', 20 * '-', "Finished Training LogisticRegressionWithLBFGS", 20 * '-'


        test = test.map(lambda l: (l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7])).map(logistic_train).cache()
        #print test.take(100)


        labelsAndPreds = test.map(lambda p: (p.label, LR_model.predict(p.features)))
        trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(test.count())
        # Computing Recall
        print("Training Error = " + str(trainErr))
        print "\n Recall = " + str(1-trainErr)
        # print predictions.collect()

    sc.stop()
