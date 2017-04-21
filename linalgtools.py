#!/usr/bin/python

import numpy as np
import scipy as sp
from scipy import linalg
import math
import sys

global eps
eps = 1E-8

def null(A):
	u, s, vh = sp.linalg.svd(A, full_matrices=True)

	if np.linalg.norm(np.dot(A, np.transpose(vh[len(s):])), 'fro') > eps :
		print 'Error: Nullspace not accurate'
		#sys.exit(2)
		return None

	return vh[len(s):]

def findrays(Abasic):
	#Find extreme ray directions of the simplicial cone defined by Abasic x <= bbasic
	
	dirs = -np.transpose(np.linalg.inv(Abasic))
	
	return dirs
