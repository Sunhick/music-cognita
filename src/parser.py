def parse_id(userid):
    # from some reason I have to use 7 'f's instead of 8 to get the
    # right number of bits or else spark will complain about not
    # being able to cast from a long to an integer.
    return {'id': userid, 'hash': int(hash(userid) & 0xfffffff)}

def parse_line(line):
    line = line.split()
    user = parse_id(line[0])
    song = parse_id(line[1])
    rating = float(line[2])
    return {'user':user, 'song':song, 'rating':rating}

def parseForRandomForest(line):
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

global ff
ff = dict({u'jazz and blues':1, u'classic pop and rock':2, u'classical':3, u'punk':4, u'metal':5, u'pop':6, u'dance and electronica':7, u'hip-hop':8, u'soul and reggae':9, u'folk':0})
def parse(line):
	line = line.split(',')
	genre = line[0]
	gid = ff[genre]
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
	return LabeledPoint(gid, Vectors.dense(features))