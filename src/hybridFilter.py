__author__ = "Sunil"
__copyright__ = "Copyright 2016"
__license__ = "MIT License"
__version__ = "0.1.0"
__email__ = "suba5417@colorado.edu"

import os
import argparse
import pyspark as ps
import numpy as np

from baseFilter import Filter

class HybridFilter(Filter):
	def __init__(self):
		pass

def main(args):
	hfilter = HybridFilter()

if __name__ == '__main__':
	parser  = argparse.ArgumentParser(description='Hybrid filtering for music recommender')
	parser.add_argument('--train', type=str, help='python hybridFilter.py <path_to_dataset>')
	parser.add_argument('--test',  type=str, help='python hybridFilter.py <path_to_dataset>')
	parser.add_argument('-v', '--verbose', action='count', help='verbosity')

	args = parser.parse_args()
	main()