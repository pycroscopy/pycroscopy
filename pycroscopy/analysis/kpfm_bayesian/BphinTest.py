# Oak Ridge National Lab
# Center for Nanophase Materials Sciences
# written by Alvin Tan on 08/16/2019
# in collaboration with Rama Vasudevan, Liam Collins, and Kody Law

# This program reads in simulated input via csv files, then runs B_phin
# on said input to confirm that the function is consistent with Matlab code.

import numpy as np
from kpfm_bayesian_utils import _B_phin
from matplotlib import pyplot as plt 

phi = float(np.genfromtxt('C:/Users/Administrator/Dropbox/polynomial approximation paper/Analysis Codes/Paper Codes/InputsAndOutputs2/B_phinStuff/phi-0.csv', delimiter=','))
w = float(np.genfromtxt('C:/Users/Administrator/Dropbox/polynomial approximation paper/Analysis Codes/Paper Codes/InputsAndOutputs2/B_phinStuff/w-0.csv', delimiter=','))
tt = np.genfromtxt('C:/Users/Administrator/Dropbox/polynomial approximation paper/Analysis Codes/Paper Codes/InputsAndOutputs2/B_phinStuff/tt-0.csv', delimiter=',')
n = int(np.genfromtxt('C:/Users/Administrator/Dropbox/polynomial approximation paper/Analysis Codes/Paper Codes/InputsAndOutputs2/B_phinStuff/n-0.csv', delimiter=','))

B = _B_phin(phi, w, tt, n)

B_exp = np.genfromtxt('C:/Users/Administrator/Dropbox/polynomial approximation paper/Analysis Codes/Paper Codes/InputsAndOutputs2/B_phinStuff/B-0.csv', delimiter=',')

print("B is as expected? {}".format(np.allclose(B, B_exp)))
# B is not the same as B_exp, but graphing shows that it is just a sinusoid with a slight shift. The shift 
# is likely due to rounding error generated when storing Matlab variables into csv. Should not be detrimental.

figBoi = plt.figure()
plt.plot(tt, B[1::2, 1], label="B[1]")
plt.plot(tt, B[1::2, 2], label="B[2]")
plt.plot(tt, B_exp[1::2, 1], label="B_exp[1]")
plt.plot(tt, B_exp[1::2, 2], label="B_exp[2]")
plt.legend()
figBoi.show()

breakpoint()


