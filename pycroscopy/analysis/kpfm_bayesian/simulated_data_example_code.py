# Oak Ridge National Lab
# Center for Nanophase Materials Sciences
# written by Alvin Tan on 08/15/2019
# in collaboration with Rama Vasudevan, Liam Collins, and Kody Law

# This program reads in simulated input via csv files, then runs Bayesian inference
# on said input to confirm that the processes are functional.

import numpy as np
from .kpfm_bayesian_utils import get_default_parameters, process_pixel

# Grab the simulated data.
# Note: simulated data was produced by Matlab code in RunSimulations_SNRnoiseLooping.m and saved
# in order to compare the results of the Matlab code with that of this Python code.
R_H = np.genfromtxt('simulated_data/R_H.csv', delimiter=',')
wd = int(np.genfromtxt('simulated_data/wd.csv', delimiter=','))
n0 = int(np.genfromtxt('simulated_data/n0.csv', delimiter=','))
p = get_default_parameters()
Rforce = np.genfromtxt('simulated_data/Rforce.csv', delimiter=',')

# Then run the Bayesian inference on the simulated data. We then display the graphs and save them.
# This is identical to creating a KPFM Process class and runnint test(verbose=True) if you had an h5 file.
resultGraphs = process_pixel(R_H, wd, p, Rforce=Rforce, graph=True, verbose=True)
resultGraphs[0].savefig("simulated_data/3Dplot.png")
resultGraphs[1].savefig("simulated_data/OtherPlots.png")

# Pretty self explanitory here.
input("Pause here to inspect the graphs. Press <Enter> to exit the program...")