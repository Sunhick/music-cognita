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


def logistic_test(line):
    dance = float(line[1])*10
    energy = float(line[2])*10
    key = float(line[3])
    loudness = float(line[4])
    tempo = float(line[5])
    time_sig = int(line[6])
    track_id = line[0]

    features = [dance, energy, key, loudness, tempo, time_sig]

    return (track_id, Vectors.dense(features))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A Hybrid Filter to recommend songs')
    parser.add_argument('-d', '--datapath', help='Path where data is stored Apache Parquet format', required=True)
    parser.add_argument('-m', '--modelpath', help='Path to where the ALS model was saved', required=True)
    args = parser.parse_args()
    database = args.datapath

    conf = SparkConf().setAppName("Hybrid filter").set("spark.executor.memory", '3g')
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

    # Get all the song details
    user_played_songs_details = played_songs.join(df_songs, df_songs.song_id == played_songs.song_id) \
        .select(df_songs.song_id, "title", "play_count", "artist_name", "duration", "year", "track_id") \
        .dropDuplicates(['title', 'artist_name']).cache()
    user_played_songs_details = user_played_songs_details.sort(user_played_songs_details.play_count.desc()).cache()

    # Show the played details
    print "\n User listening history\n"
    user_played_songs_details.show(30, truncate=True)

    # Run ALS Algorithm
    print 20 * '-', "Running ALS model", 20 * '-'
    user_songs_id = played_songs.select(df_triplets.song_id).cache()

    # Convert Song id to hash id for ALS model

    ALS_predictions = ALSmodel.recommendProducts(user_hash, 1000)
    # print ALS_predictions

    max_ALS_rating = ALS_predictions[0].rating
    print '\n', 20 * '-', "ALS model finished", 20 * '-'

    # Convert to RDD and normalize rating from 0 to 10
    df_ALS_predictions = sc.parallelize(ALS_predictions) \
        .map(lambda l: Rating(l.user, l.product, l.rating / max_ALS_rating * 10)) \
        .toDF().cache()

    # df_ALS_predictions.show()

    # Convert df_ALS_predictions which has hash values into id values
    df_ALS_songs = df_ALS_predictions.join(df_song_hash_id, df_ALS_predictions.product == df_song_hash_id.hash_id) \
        .select(df_song_hash_id.song_id, df_ALS_predictions.rating).dropDuplicates().cache()

    df_ALS_songs = df_ALS_songs.withColumnRenamed('rating', 'predicted_score').cache()

    # df_ALS_songs.sort(df_ALS_songs.song_id.desc()).show(100)

    # Get attributes of the songs predicted by ALS
    print '\n', 20 * '-', "Top 20 Songs Recommended by Collaborative Filter(ALS)", 20 * '-', '\n'
    df_ALS_songs = df_ALS_songs.join(df_songs, df_songs.song_id == df_ALS_songs.song_id)\
        .dropDuplicates(['title', 'artist_name']).drop(df_ALS_songs.song_id).cache()

    ALS_recommend = df_ALS_songs.select(df_ALS_songs.track_id, "title", "artist_name", "duration", "year", df_ALS_songs.predicted_score)\
                        .sort(df_ALS_songs.predicted_score.desc()).cache()

    ALS_recommend.show(20)


    # Step two: Find songs similar to user listened songs

    played_tracks = user_played_songs_details.select("track_id", "play_count")
    # print played_tracks

    similar_details = df_song_sim.join(played_tracks, played_tracks.track_id == df_song_sim.tid). \
        select(df_song_sim.tid, df_song_sim.target, played_tracks.play_count).dropDuplicates() \
        .sort(df_song_sim.tid) \
        .map(lambda l: (l[0], l[1], l[2])).collect()

    # print similar_details

    similar_songs = defaultdict(float)
    for i in xrange(len(similar_details)):
        parent_track_id = similar_details[i][0]
        parent_track_count = similar_details[i][2]
        track_data = similar_details[i][1].split(",")

        # print parent_track_id, parent_track_count, track_data
        for j in xrange(len(track_data)):
            if j % 2 == 0:
                similar_songs[track_data[j]] += (float(track_data[j + 1]) * parent_track_count)

    # Sort the similarity results
    top_similarity = sorted(similar_songs.items(), key=lambda x: x[1], reverse=True)[:50]

    max_score = top_similarity[0][1]

    # print top_similarity
    df_top_similarity = sc.parallelize(top_similarity).map(lambda l: (l[0], l[1] * 10 / max_score)).cache()
    # print df_top_similarity.collect()

    df_top_similarity = df_top_similarity.toDF(["track_id", "predicted_score"]).cache()

    # Songs recommended by similarity matching
    print '\n', 20 * '-', 'Songs recommended by matching similar songs', 20 * '-'

    SIM_recommend = df_top_similarity.join(df_songs, df_songs.track_id == df_top_similarity.track_id) \
        .select(df_songs.track_id, "title", "artist_name", "duration", "year", "predicted_score")\
        .dropDuplicates(['title', 'artist_name']).cache()

    SIM_recommend = SIM_recommend.sort(SIM_recommend.predicted_score.desc()).cache()


    SIM_recommend.show(20)

    # Predict recommendation using Logistic regression. Pass ALS output here

    df_ALS_attributes = df_ALS_songs.filter(df_ALS_songs.year <= 2009).select(df_ALS_songs.track_id, "danceability",
                        "energy", "key", "loudness", "tempo", "time_signature", "predicted_score")\
                        .where(df_ALS_songs.predicted_score > 0).cache()

    train_data_from_ALS = df_ALS_attributes.map(lambda l: (l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7])).sortBy(lambda x:x[7]).map(
        logistic_train).cache()

    #print train_data_from_ALS.take(1000)

    print '\n', 20 * '-', "Start Training LogisticRegressionWithLBFGS using ALS predicted songs", 20 * '-'

    #number of class = 11 because ratings can be 0,1,2,3,4....,9,10
    LR_model = LogisticRegressionWithLBFGS.train(train_data_from_ALS, numClasses=11)

    print '\n', 20 * '-', "Finished Training LogisticRegressionWithLBFGS", 20 * '-'

    df_new_songs = df_songs.filter(df_songs.year > 2009).select(df_songs.track_id, "danceability", "energy", "key",
                                   "loudness", "tempo", "time_signature").cache()

    test = df_new_songs.map(lambda l: (l[0], l[1], l[2], l[3], l[4], l[5], l[6])).map(logistic_test).cache()
    #print test.take(100)

    print '\n', 20 * '-', 'Predicting Recommendations using Logistic Regression', 20 * '-', '\n'
    predictions = test.map(lambda p: (p[0], LR_model.predict(p[1]))).cache()

    # print predictions.collect()

    LR_songs_rdd = sc.parallelize(predictions.collect())
    df_LR_songs = LR_songs_rdd.toDF(["track_id", "predicted_score"])
    df_LR_songs = df_LR_songs.select(df_LR_songs.track_id, df_LR_songs.predicted_score)\
                    .dropDuplicates().sort(df_LR_songs.predicted_score.desc()).cache()

    df_recomm = df_LR_songs.join(df_songs, df_LR_songs.track_id == df_songs.track_id) \
        .select(df_songs.track_id, "title", "artist_name", "duration", "year", df_LR_songs.predicted_score)\
        .dropDuplicates(['title', 'artist_name']).cache()

    df_recomm = df_recomm.select(df_recomm.track_id, "title", "artist_name", "duration", "year", df_recomm.predicted_score)\
                        .dropDuplicates().cache()
    df_recomm = df_recomm.sort(df_recomm.predicted_score.desc())

    df_recomm.show(20)

    print '\n', 20 * '-', 'Your Final Recommendations are', 20 * '-', '\n'

    all_recommend = ALS_recommend.unionAll(SIM_recommend).unionAll(df_recomm).cache()
    all_recommend.sort(all_recommend.predicted_score.desc()).show(40)

    sc.stop()
