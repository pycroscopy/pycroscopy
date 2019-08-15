# Oak Ridge National Lab
# Center for Nanophase Materials Sciences
# written by Alvin Tan on 08/06/2019
# in collaboration with Rama Vasudevan, Liam Collins, and Kody Law

# This program provides the Pycroscopy framework to process messy, noisy
# data streamed in from a fundamentally new data acquisition method
# incepted by Liam and involving Kelvin probe microscopy.

import scipy.optimize as spo
import scipy.linalg as spla
import numpy as np
import time

def process_pixel():
	# TODO
	return


def get_default_parameters():
	'''
	Returns default parameter dictionary for an NC-AFM setup
	'''

	p = {}

	# detection and excitation system parameters
	p['SYS.dzds'] = 81e-15 	# Detection system noise 
							# amplitude spectral density d_ds^z in m/sqHz
	             
	p['SYS.INVOLS'] = 52.5e-9	# Sensitivity of detection system in m/V

	# Cantilever parameters
	p['CL.f0'] = 58e3 		# resonance frequency in Hz
	p['CL.Q'] = 115 		# quality factor (unitless)
	p['CL.k'] = 2.8 		# cantilever stiffness in N/m
	p['CL.A'] = 1 			# amplitude (zero-peak) in m
	p['CL.T'] = 273.15+23.8 # CL temperature in K
	p['CL.f1'] = 350e3 		# 2nd Eig resonance frequency in Hz
	p['CL.Q1'] = 350 		# 2nd Eig quality factor (unitless)
	p['CL.A1'] = 0.0000 	# 2nd Eig amplitude (zero-peak) in m

	# Tip geometry
	p['Tip.c_angle'] = 32*np.pi/180 	# Cone angle of probe
	p['TS.Rtip'] = 20e-9 				# Radius of probe apex (m)
	p['TS.Rheight'] = 22e-6 			# Height of probe (i.e. distance from tip apex to cantilever) (m)
	p['TS.eps_z'] = 8.85e-12 			# permitivity


	# tip-sample interaction
	p['TS.alpha'] = -12.3e9 # alpha_ts in Hz/m
	p['TS.beta'] = 12.3e9 	# beta_ts in Hz/m

	# Data Simulation Parameters
	p['Sim.Tmax'] = 4.096e-3*2 	# simulation time (s)
	p['Sim.IOrate'] = 4e6 		# Sampling rate (Hz)
	p['Sim.Vfreq'] = 56e3 		# AC voltage Drive Frequency (Hz)
	p['Sim.VAmp'] = 3 			# AC voltage Drive Amplitude (p2p)
	p['Sim.VCPD'] = -1			# AC voltage Drive Amplitude (p2p)
	p['Sim.VDC'] = 0
	p['Sim.Phasshift'] = 0.2 	## Some noise
	p['Sim.NAmp'] = 1e-6 		## Some noise
	p['Sim.snr'] = 12

	# Bayesian Parameters
	p['Bayes.fac'] = 128 	# factor to reduce length of timeseries
	p['Bayes.Npoly'] = 2
	p['Bayes.aa'] = 2 		##hyperparameters

	return p


def _B_phin(phi, w, tt, n):
	'''
	phi is a number
	w is a number
	tt is a numpy column vector
	n is probably a number
	'''
	N = tt.size
	Bn = np.zeros((2*N, n+1))

	#breakpoint()

	for i in range(n+1):
		Bn[1:2*N+1:2, i] = np.squeeze(np.sin(w*tt + phi)**i)

	return Bn


def _mmlenn(pp, w, tt, AA, a1, a2, y, n, M, h, m0, sigi, aa):
	'''
	pp is a row vector
	w is a constant
	tt is a column vector
	AA is a square matrix
	a1 and a2 are unclear
	y is a numpy column vector
	n is a number?
	M is a number?
	h is a number
	m0 is a column vector
	sigi is probably a number
	aa is something
	'''
	#breakpoint()

	#print("pp is {}".format(pp))

	phi = pp[2]
	sig = pp[0]
	gam = pp[1]
	N = y.size
	P0 = np.diag(np.concatenate((np.array([1/sigi/sigi, 1/sigi/sigi]), 1/sig/sig*(np.arange(1, n+2)**aa))))
	C0 = np.linalg.inv(P0)
	GAI = 1/gam/gam*np.eye(N)

	B = _B_phin(phi, w, tt, n)
	BB = np.matmul(AA, B)*h
	CC1 = np.concatenate((a1.reshape((a1.size, 1), order='F'),
						  a2.reshape((a2.size, 1), order='F'), BB), axis=1)
	CC = CC1[::2, :]

	# matrix multiplication? may be problematic
	Sig, resid, rank, s = np.linalg.lstsq(P0 + np.matmul(np.matmul(CC.T, GAI), CC), np.eye(M), rcond=None)
	m_phi = np.matmul(Sig, np.matmul(np.matmul(CC.T, GAI), y) + np.matmul(P0, m0))

	if (-np.pi <= phi) and (phi <= 0):
		# matrix multiplication? may be problematic
		out = np.sum(np.log(gam**2*np.ones((N, 1)))) + np.sum(np.log(np.linalg.eig(Sig)[0])) + \
			  np.matmul(np.matmul(y.T, GAI), y) - np.matmul(np.matmul(np.matmul(Sig, CC.T), np.matmul(GAI, y)).T, np.matmul(np.matmul(CC.T, GAI), y)) + 0*sig**2/200 # what is the point of this 0*...
		out = float(out)
	else:
		out = np.inf

	#breakpoint()

	#print("out is {}".format(out))

	return out


def BayesianInference(R_H, wd, n0, p):
	'''
	R_H is a numpy row vector
	wd is a number
	n0 is a number
	p is a dictionary
	'''
	fac = p["Bayes.fac"]
	Qi = 1/p["CL.Q"]
	OmV = wd*(1e-3)
	Om0 = p["CL.f0"]*(1e-3)
	w = OmV/Om0
	L = np.array([[0, 1], [-1, -Qi]])
	wr1 = p["CL.f0"]

	N = int(R_H.size/fac)

	T=p["Sim.Tmax"]*wr1*2*np.pi/fac

	R_H = R_H - np.mean(R_H)

	#breakpoint()

	y = (1e9)*R_H[:N][np.newaxis].T # may get messed up
	A = np.zeros((2, 2*N))
	AA = np.zeros((2*N, 2*N))
	h = T/N
	tt = np.arange(T, step=h)[np.newaxis].T # inclusive of T-h, may get messed up

	startTime = time.time()
	for i in range(N):
		A[:, 2*i:2*(i+1)] = spla.expm(L*h*(i+1)) # may want to change this to L*h*(i+1)

	a1 = np.fliplr(A[:, ::2])
	a2 = np.fliplr(A[:, 1::2])
	A1 = 0*A
	A1[:, ::2] = a1
	A1[:, 1::2] = a2

	for j in range(1, N):
		AA[2*j:2*(j+1), :2*j] = A1[:, -2*j:]

	a1 = A[:, ::2]
	a2 = A[:, 1::2]

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

	#breakpoint()

	m0 = np.concatenate((np.array([y[0] - (y[1]-y[0])]),
						 np.array([(y[1] - y[0])/h]),
						 np.zeros((n+1, 1))),
						axis=0)

	startTime = time.time()

	B = _B_phin(phi, w, tt, n)
	BB = np.matmul(AA, B)*h

	#breakpoint()

	CC1 = np.concatenate((a1.reshape((a1.size, 1), order='F'),
						  a2.reshape((a2.size, 1), order='F'), BB), axis=1)
	CC = CC1[::2, :]

	# This may need matrix multiplication.
	# Use np.linalg.lstsq for Matlab's left divide
	Sig, resid, rank, s = np.linalg.lstsq(P0 + np.matmul(np.matmul(CC.T, GAI), CC), np.eye(M), rcond=None)
	m_phi = np.matmul(Sig, np.matmul(np.matmul(CC.T, GAI), y) + np.matmul(P0, m0))

	rrmse = np.linalg.norm(y - np.matmul(CC, m_phi))/np.linalg.norm(y)

	forward_time = time.time() - startTime

	# Optimize hyperparameters
	sd = 1
	np.random.seed(sd)

	pp0 = np.zeros(3)
	pp0[:2] = np.random.randn(1, 2)**2 
	pp0[2] = -np.pi*np.random.rand()

	# Blackbox optimization over Phi(theta; y)
	startTime = time.time()

	#breakpoint()
	#pp1, fval, exitflag = fminsearch(@(pp)_mmlenn(pp, w, tt, AA, a1, a2, y, n, M, h, m0, sigi, aa), pp0) # TODO
	#pp1, fval, numIter = spo.fmin(lambda pp: _mmlenn(pp, w, tt, AA, a1, a2, y, n, M, h, m0, sigi, aa), pp0)

	print("about to execute first call to _mmlenn. pp is pp0 = {}".format(pp0))
	optResult = spo.minimize(lambda pp: _mmlenn(pp, w, tt, AA, a1, a2, y, n, M, h, m0, sigi, aa), pp0)
	print("finished first call to _mmlenn")

	#breakpoint() # The mmlenn function may prove problematic... also on line 248

	optim_time_mle = time.time() - startTime

	fvalg = optResult.fun
	pp1g = optResult.x

	for i in range(10):
		pp0[:2] = np.random.randn(1, 2)**2
		pp0[2] = -np.pi*np.random.rand()

		startTime = time.time()
		#options = optimset("MaxIter", 1e12, "TolFun", 1e-18, "TolX", 1e-18) # TODO
		#pp1, fval, exitflag = fminsearch(@(pp)_mmlenn(pp, w, tt, AA, a1, a2, y, n, M, h, m0, sigi, aa), pp0, options) # TODO
		#pp1, fval, numIter = spo.fmin(lambda pp: _mmlenn(pp, w, tt, AA, a1, a2, y, n, M, h, m0, sigi, aa), pp0,
		#					 maxiter=int(1e12), ftol=1e-18, xtol=1e-18)
		optResult = spo.minimize(lambda pp: _mmlenn(pp, w, tt, AA, a1, a2, y, n, M, h, m0, sigi, aa), pp0,
								 options={"maxiter":int(1e12)}, tol=1e-18)
		optim_time_mle = time.time() - startTime
		print("pp1 is {}".format(optResult.x))
		print("fval is {}".format(optResult.fun))
		if optResult.fun < fvalg:
			pp1g = optResult.x
			fvalg = optResult.fun

	pp1 = pp1g

	startTime = time.time()

	phi = pp1[2]
	sig = pp1[0]
	gam = pp1[1]
	B = _B_phin(phi, w, tt, n)
	BB = np.matmul(AA, B)*h 
	CC1 = np.concatenate((a1.reshape((a1.size, 1), order='F'),
						  a2.reshape((a2.size, 1), order='F'), BB), axis=1)
	CC = CC1[::2, :]
	P0 = np.diag(np.concatenate((np.array([1/sigi/sigi, 1/sigi/sigi]), 1/sig/sig*(np.arange(1, n+2)**aa))))

	C0 = np.linalg.inv(P0)
	GAI = 1/gam/gam*np.eye(N)

	# matrix multiplication? may be problematic
	Sig, resid, rank, s = np.linalg.lstsq(P0 + np.matmul(np.matmul(CC.T, GAI), CC), np.eye(M), recond=None)
	m_phi = np.matmul(Sig, np.matmul(np.matmul(CC.T, GAI), y) + np.matmul(P0, m0))
	rrmse = np.linalg.norm(y - np.matmul(CC, m_phi))/np.linalg.norm(y)
	print(time.time() - startTime)

	return y, tt, pp1, sig, gam, AA, B, BB, CC, C0, P0, CC1, GAI, M, m0, phi, m_phi, Sig
















