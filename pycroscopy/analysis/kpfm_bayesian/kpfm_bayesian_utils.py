# Oak Ridge National Lab
# Center for Nanophase Materials Sciences
# written by Alvin Tan on 08/06/2019
# in collaboration with Rama Vasudevan, Liam Collins, and Kody Law

# This program provides the Pycroscopy framework to process messy, noisy
# data streamed in from a fundamentally new data acquisition method
# incepted by Liam and involving Kelvin probe microscopy.

import scipy.linalg as spla
import numpy as 
import math
import time

def process_pixel():
	# TODO
	return

def BayesianInference(R_H, wd, n0, p):
	'''
	R_H is a numpy vector
	wd
	n0
	p is a dictionary
	'''
	fac = p["Bayes.fac"]
	Qi = 1/p["CL.Q"]
	OmV = wd*(1e-3)
	Om0 = p["CL.f0"]*(1e-3)
	w = OmV/Om0
	L = np.array([[0, 1], [-1, -Qi]])
	wr1 = p["CL.f0"]

	N = R_H.size/fac

	T=p["Sim.Tmax"]*wr1*2*math.pi/fac

	R_H = R_H - np.mean(R_H)
	y = R_H[:N]*(1e9).T # may get messed up
	A = np.zeros((2, 2*N))
	AA = np.zeros((2*N, 2*N))
	h = T/N
	tt = np.arange(T, h).T # inclusive of T-h, may get messed up

	startTime = time.time()
	for i in range(N):
		A[:, 2*i:2*(i+1)] = spla.expm(L*h*i)

	a1 = np.fliplr(A[:, ::2])
	a2 = np.fliplr(A[:, 1::2])
	A1 = 0*A
	A1[:, ::2] = a1
	A1[:, 1::2] = a2

	for j in range(1, N):
		AA(2*j:2*(j+1), :2*j) = A1(:, -2*j:)

	a1 = A(:, ::2)
	a2 = A(:, 1::2)

	prelim_time = time.time() - startTime

	# Hyperparameters
	n = p["Bayes.Npoly"]
	M = 2+n+1;
	sigi = 3;
	sig = 1;
	gam = 0.1
	phi = 1
	aa = p["Bayes.aa"]

	P0 = np.diag(np.concatenate((np.array([1/sigi/sigi, 1/sigi/sigi]), 1/sig/sig*(np.arange(1, n+2)**aa))))
	C0 = np.linalg.inv(P0)
	GAI = 1/gam/gam*np.eye(N)
	m0 = np.concatenate((y[0] - (y[1]-y[0]),
						 (y[1] - y[0])/h,
						 np.zeros((n+1, 1))),
						axis=0)

	startTime = time.time()

	B = B_phin(phi, w, tt, n) # TODO
	BB = AA*B*h # Are these matrices? may be problematic
	CC1 = np.concatenate((a1, a2, BB), axis=1)
	CC = CC1(::2, :)

	# This may need matrix multiplication.
	# Use np.linalg.lstsq for Matlab's left divide
	Sig, resid, rank, s = np.linalg.lstsq(P0 + CC.T * GAI * CC, np.eye(M))
	m_phi = Sig*(CC.T * GAI * y + P0 * m0)

	rrmse = np.linalg.norm(y-CC*m_phi)/np.linalg.norm(y)

	forward_time = time.time() - startTime

	# Optimize hyperparameters
	sd = 1
	np.random.seed(sd)

	pp0[:2] = np.random.randn(2, 1)**2 # is pp0 a column vector? may be problematic
	pp0[2] = -math.pi*np.random.rand()

	# Blackbox optimization over Phi(theta; y)
	startTime = time.time()
	pp1, fval, exitflag = fminsearch(@(pp)mmlen(pp, w, tt, AA, a1, a2, y, n, M, h, m0, sigi, aa), pp0) # TODO
	optim_time_mle = time.time() - startTime

	fvalg = fval
	pp1g = pp1

	for i in range(10):
		pp0(:2) = np.random.randn(2, 1)**2
		pp0(2) = -math.pi*np.random.rand()

		startTime = time.time()
		options = optimset("MaxIter", 1e12, "TolFun", 1e-18, "TolX", 1e-18) # TODO
		pp1, fval, exitflag = fminsearch(@(pp)mmlenn(pp, w, tt, AA, a1, a2, y, n, M, h, m0, sigi, aa), pp0, options) # TODO
		optim_time_mle = time.time() - startTime
		print("pp1 is {}".format(pp1))
		print("fval is {}".format(fval))
		if fval < fvalg:
			pp1g = pp1
			fvalg = fval

	pp1 = pp1g

	startTime = time.time()

	phi = pp1[2]
	sig = pp1[0]
	gam = pp1[1]
	B = B_phin(phi, w, tt, n) # TODO
	BB = AA*B*h # matrices? may be problematic
	CC1 = np.concatenate((a1, a2, BB), axis=1)
	CC = CC1[::2, :]
	P0 = np.diag(np.concatenate((np.array([1/sigi/sigi, 1/sigi/sigi]), 1/sig/sig*(np.arange(1, n+2)**aa))))

	C0 = np.linalg.inv(P0)
	GAI = 1/gam/gam*np.eye(N)

	# matrix multiplication? may be problematic
	Sig, resid, rank, s = np.linalg.lstsq(P0 + CC.T * GAI * CC, np.eye(M))
	m_phi = Sig*(CC.T * GAI * y + P0*m0)
	rrmse = np.linalg.norm(y-CC*m_phi)/np.linalg.norm(y)
	print(time.time() - startTime)


















