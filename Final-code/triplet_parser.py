def hash_id(id):
    #create a Hash of given IDs.
    return {'id': id, 'hash': int(hash(id) & 0xfffffff)}

def parse_line(line):
    parserChar = '\t'
    line = line.split(parserChar)
    user = hash_id(line[0])
    song = hash_id(line[1])
    rating = float(line[2])
    return {'user': user, 'song': song, 'rating': rating}

def format_triplets(input_rdd):
    triplet_in_dict_form = input_rdd.map(parse_line)
    data_triplet = triplet_in_dict_form.map(lambda x: (x['user']['hash'], x['song']['hash'], x['rating']))
    return (triplet_in_dict_form, data_triplet)
