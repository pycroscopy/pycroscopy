# Oak Ridge National Lab
# Center for Nanophase Materials Sciences
# written by Alvin Tan on 08/06/2019
# in collaboration with Rama Vasudevan, Liam Collins, and Kody Law

# This program provides the Pycroscopy framework to process messy, noisy
# data streamed in from a fundamentally new data acquisition method
# incepted by Liam and involving Kelvin probe microscopy.

from __future__ import division, print_function, absolute_import, unicode_literals
from pyUSID.processing.process import Process

# Helper function for importing packages if they don't exist
import os
import subprocess
import sys
def install(package):
	subsystem.call([sys.executable, "-m", "pip", "install", "--user", package])

try:
	import pyUSID as usid 
except ImportError:
	print("pyUSID not found. Will install with pip.")
	import pip
	install("pyUSID")
	import pyUSID as usid 

try:
	import pycroscopy as px 
except ImportError:
	print("pycroscopy not found. Will install with pip.")
	import pip
	install("pycroscopy")
	import pycroscopy as px 

from pyUSID.processing.process import Process

class KPFMBayesianInference(Process):
	def __init__(self, h5_main, **kwargs):
		'''
		Description of functions, inputs, and outputs
		'''
		super(KPFMBayesianInference, self).__init__(h5_main, **kwargs)

		










