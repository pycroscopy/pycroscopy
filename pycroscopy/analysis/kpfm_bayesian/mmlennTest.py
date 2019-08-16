# Oak Ridge National Lab
# Center for Nanophase Materials Sciences
# written by Alvin Tan on 08/16/2019
# in collaboration with Rama Vasudevan, Liam Collins, and Kody Law

# This program reads in simulated input via csv files, then runs mmlenn
# on said input to confirm that the function is consistent with Matlab code.

import numpy as np
from kpfm_bayesian_utils import _mmlenn
from matplotlib import pyplot as plt 
import scipy.optimize as spo

pp0 = np.genfromtxt('C:/Users/Administrator/Dropbox/polynomial approximation paper/Analysis Codes/Paper Codes/InputsAndOutputs2/mmlennStuff/pp0-1.csv', delimiter=',')
w = float(np.genfromtxt('C:/Users/Administrator/Dropbox/polynomial approximation paper/Analysis Codes/Paper Codes/InputsAndOutputs2/mmlennStuff/w-1.csv', delimiter=','))
tt = np.genfromtxt('C:/Users/Administrator/Dropbox/polynomial approximation paper/Analysis Codes/Paper Codes/InputsAndOutputs2/mmlennStuff/tt-1.csv', delimiter=',')
AA = np.genfromtxt('C:/Users/Administrator/Dropbox/polynomial approximation paper/Analysis Codes/Paper Codes/InputsAndOutputs2/AA-1.csv', delimiter=',')
a1 = np.genfromtxt('C:/Users/Administrator/Dropbox/polynomial approximation paper/Analysis Codes/Paper Codes/InputsAndOutputs2/mmlennStuff/a1-1.csv', delimiter=',')
a2 = np.genfromtxt('C:/Users/Administrator/Dropbox/polynomial approximation paper/Analysis Codes/Paper Codes/InputsAndOutputs2/mmlennStuff/a2-1.csv', delimiter=',')
y = np.genfromtxt('C:/Users/Administrator/Dropbox/polynomial approximation paper/Analysis Codes/Paper Codes/InputsAndOutputs2/mmlennStuff/y-1.csv', delimiter=',')
n = int(np.genfromtxt('C:/Users/Administrator/Dropbox/polynomial approximation paper/Analysis Codes/Paper Codes/InputsAndOutputs2/mmlennStuff/n-1.csv', delimiter=','))
M = int(np.genfromtxt('C:/Users/Administrator/Dropbox/polynomial approximation paper/Analysis Codes/Paper Codes/InputsAndOutputs2/mmlennStuff/M-1.csv', delimiter=','))
h = np.genfromtxt('C:/Users/Administrator/Dropbox/polynomial approximation paper/Analysis Codes/Paper Codes/InputsAndOutputs2/mmlennStuff/h-1.csv', delimiter=',')
m0 = np.genfromtxt('C:/Users/Administrator/Dropbox/polynomial approximation paper/Analysis Codes/Paper Codes/InputsAndOutputs2/mmlennStuff/m0-1.csv', delimiter=',')
sigi = np.genfromtxt('C:/Users/Administrator/Dropbox/polynomial approximation paper/Analysis Codes/Paper Codes/InputsAndOutputs2/mmlennStuff/sigi-1.csv', delimiter=',')
aa = np.genfromtxt('C:/Users/Administrator/Dropbox/polynomial approximation paper/Analysis Codes/Paper Codes/InputsAndOutputs2/mmlennStuff/aa-1.csv', delimiter=',')


optResult = spo.minimize(lambda pp: _mmlenn(pp, w, tt, AA, a1, a2, y, n, M, h, m0, sigi, aa), pp0)

fval_exp = np.genfromtxt('C:/Users/Administrator/Dropbox/polynomial approximation paper/Analysis Codes/Paper Codes/InputsAndOutputs2/mmlennStuff/fval-1.csv', delimiter=',')

print("fval is as expected? {}".format(np.allclose(optResult.fun, fval_exp)))

breakpoint()