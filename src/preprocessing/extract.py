"""
extract.py
"""

__author__ = "Sunil"
__copyright__ = "Copyright 2016"
__license__ = "MIT License"
__version__ = "0.1.0"
__email__ = "suba5417@colorado.edu"

import os
import sys
import hdf5_getters
import numpy as np
import argparse
import csv

from fnmatch import fnmatch

# To keep track of the % of progess. It's hard-coded for performance reasons.
# Roughly 10k entries for subset of dataset & ~1 million entries in full dataset
kCount = 10001

""" 

Understanding audio_summary results
The tempo field represents the BPM of the song in question. In this case, it's 142 BPM. Other interesting attributes are:

* danceability: A number that ranges from 0 to 1, representing how danceable The Echo Nest thinks this song is.
* duration: Length of the song, in seconds.
* energy: A number that ranges from 0 to 1, representing how energetic The Echo Nest thinks this song is.
* key: The key that The Echo Nest believes the song is in. Key signatures start at 0 (C) and ascend the chromatic scale. 
	In this case, a key of 1 represents a song in D-flat.
* loudness: The overall loudness of a track in decibels (dB).
* mode: Number representing whether the song is in a minor (0) or major (1) key. Use this in conjunction with 'key'.
* time_signature: Time signature of the key; how many beats per measure.


list of all getters from h5 files

['get_danceability', 'get_song_id', 'get_release', 'get_artist_hotttnesss', 'get_title', 'get_segments_timbre', 'get_artist_longitude', 
'get_beats_confidence', 'get_end_of_fade_in', 'get_time_signature', 'get_artist_id', 'get_sections_start', 'get_mode', 'get_loudness', 
'get_artist_7digitalid', 'get_artist_terms_freq', 'get_similar_artists', 'get_artist_terms_weight', 'get_mode_confidence', 
'get_segments_loudness_max_time', 'get_artist_familiarity', 'get_song_hotttnesss', 'get_time_signature_confidence', 'get_artist_name', 
'get_key', 'get_artist_playmeid', 'get_artist_mbtags', 'get_analysis_sample_rate', 'get_year', 'get_key_confidence', 'get_artist_location', 
'get_tatums_start', 'get_audio_md5', 'get_bars_start', 'get_bars_confidence', 'get_artist_mbid', 'get_track_7digitalid', 'get_artist_terms', 
'get_segments_pitches', 'get_segments_confidence', 'get_segments_loudness_start', 'get_energy', 'get_segments_start', 
'get_segments_loudness_max', 'get_duration', 'get_artist_mbtags_count', 'get_release_7digitalid', 'get_tatums_confidence', 'get_track_id', 
'get_tempo', 'get_start_of_fade_out', 'get_beats_start', 'get_num_songs', 'get_sections_confidence', 'get_artist_latitude']
"""

def get(getters, h5file):
	# sanity check
	if not os.path.isfile(h5file):
		print 'ERROR: file', h5file, 'does not exist.'
		sys.exit(0)
	h5 = hdf5_getters.open_h5_file_read(h5file)
	numSongs = hdf5_getters.get_num_songs(h5)
	songidx = 0
	if songidx >= numSongs:
		print 'ERROR: file contains only',numSongs
		h5.close()
		sys.exit(0)

	line = dict()
	for getter in getters:
		try:
			res = hdf5_getters.__getattribute__('get_' + getter)(h5,songidx)
		except AttributeError, e:
				print e
		if res.__class__.__name__ == 'ndarray':
			# print getter[4:]+": shape =",res.shape
			# How to put multidimensional values into file. 
			# Try to put only mean of the values etc...
			print 'Ignoring....'
		else:
			# print getter[4:]+":",res
			line[getter] = res

	h5.close()
	return line


def extract_records(root, verbose):
	pattern = "*.h5"
	txtfile = 'dataset.txt'

	with open(txtfile, 'w') as csvfile:
		fieldnames = ['song_id', 'track_id', 'artist_name', 'title', 'danceability', 'song_hotttnesss', 
				'duration', 'tempo', 'energy', 'key', 'loudness', 'time_signature']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		index = 1.

		for path, subdir, files in os.walk(root):
			h5files = [os.path.join(path, file) for file in files if fnmatch(file, pattern)]
			for h5file in h5files:
				print 'Preprocessing file ', h5file, '...\t', "[%.2f" % round((index/kCount)*100, 2), '%]'
				row = get(fieldnames, h5file)
				writer.writerow(row)
				index += 1.

def main(args):
	h5path = args.path
	verbose = args.verbose
	extract_records(h5path, verbose)

if __name__ == '__main__':
	parser  = argparse.ArgumentParser(description='Preprocessing dataset for music recommender')
	parser.add_argument('-p', '--path', type=str, help='path to HD5 files')
	parser.add_argument('-v', '--verbose', action='store_true', help='verbosity')
	parser.set_defaults(verbose=True)

	args = parser.parse_args()
	main(args)