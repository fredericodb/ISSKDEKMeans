#
# Import CWRU Matlab mat files into python
#
# http://csegroups.case.edu/bearingdatacenter/home

import scipy.io as sio
import re			# regular expression

def readmat( fname , searchkeys):
	data = sio.loadmat(fname)	# reads a python Dictorionary https://docs.python.org/2/tutorial/datastructures.html
	dic = {}

	for key in data:		# https://stackoverflow.com/questions/3294889/iterating-over-dictionaries-using-for-loops
		k = key
		for sk in searchkeys:
			matchobj = re.search( sk, k )
			if matchobj != None:
				#print 'Match: k=',k,'sk=',sk,'match=',matchobj
				dic.update({sk: data[k]})

	# return dicionary key-value pairs
	return dic