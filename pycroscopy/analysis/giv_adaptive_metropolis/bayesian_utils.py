"""
Utilities to help Adaptive Bayesian Inference computations and tests on 
USID Datasets and visualize results
Created on Tue Jul 02 2019
@author: Alvin Tan, Emily Costa
"""
import os
import time
import math
import numpy as np
import scipy.linalg as spla
from matplotlib import pyplot as plt

# Libraries for USID database use
import h5py
import pycroscopy as px 
import pyUSID as usid 

# Library to speed up adaptive metropolis
from numba import jit 

# Takes in the sine full_V wave and finds the index used to shift the waveform into
# a forward and reverse sweep. Then finds the index of the maximum value (i.e. the
# index used to split the waveform into forward and reverse sections). It returns
# both indices as well as a shifted full_V
def get_shift_and_split_indices(full_V):
    # Since we have a sine wave (which goes up, then down, then up),
    # we want to take the last bit that goes up and move it to the
    # front, such that the data first goes from -6 to 6 V, then
    # from 6 to -6 V.
    shift_index = full_V.size - 1
    while(full_V[shift_index-1] < full_V[shift_index]):
        shift_index -= 1

    full_V = np.concatenate((full_V[shift_index:], full_V[:shift_index]))

    # Then we do another walk to get the boundary between the
    # forward and reverse sections.
    split_index = 0
    while(full_V[split_index] < full_V[split_index + 1]):
        split_index += 1

    return full_V, shift_index, split_index


# Takes in a full_i_meas response and shifts it according to the given shift index
def get_shifted_response(full_i_meas, shift_index):
    return np.concatenate((full_i_meas[shift_index:], full_i_meas[:shift_index]))


# Takes a shifted array and un-shifts it according to the given shift index
def get_unshifted_response(full_i_meas, shift_index):
    new_shift_index = full_i_meas.size - shift_index
    return np.concatenate((full_i_meas[new_shift_index:], full_i_meas[:new_shift_index]))


# Takes in excitation wave amplitude and desired M value. Returns M, dx, and x.
def get_M_dx_x(V0=6, M=25):
    dx = 2*V0/(M-2)
    x = np.arange(-V0, V0+dx, dx)[np.newaxis].T
    M = x.size # M may not be the desired value but it will be very close
    return M, dx, x


# Takes in a single period of a shifted excitation wave as full_V and the corresponding
# current response as full_i_meas. Returns either the estimated resistances and 
# reconstructed currents or a pyplot figure.
def process_pixel(full_i_meas, full_V, split_index, M, dx, x, shift_index, f, V0, Ns, dvdt, pix_ind=0, graph=False, verbose=False):
    # If verbose, check if full_V and full_i_meas exist and are actually 1D
    if verbose:
        if full_V is None:
            raise Exception("full_V is None")
        if full_i_meas is None:
            raise Exception("full_i_meas is None")
        if len(full_V.shape) != 1:
            raise Exception("full_V is not one-dimensional. Its shape is {}".format(full_V.shape))
        if len(full_i_meas.shape) != 1:
            raise Exception("full_i_meas is not one-dimensional. Its shape is {}".format(full_i_meas.shape))
        if full_V.size != full_i_meas.size:
            raise Exception("full_V and full_i_meas do not have the same length. full_V is of length {} while full_i_meas is of length {}".format(full_V.size, full_i_meas.size))

    # Split up our data into forward and reverse sections
    Vfor = full_V[:split_index]
    Vrev = full_V[split_index:]
    Ifor = full_i_meas[:split_index]
    Irev = full_i_meas[split_index:]
    dvdtFor = dvdt[:split_index]
    dvdtRev = dvdt[split_index:]

    # Run the adaptive metropolis on both halves and save the results
    forward_results = _run_bayesian_inference(Vfor, Ifor, M, dx, x, f, V0, Ns, dvdtFor, verbose=verbose)
    reverse_results = _run_bayesian_inference(Vrev, Irev, M, dx, x, f, V0, Ns, dvdtRev, verbose=verbose)

    '''
    # If we want a graph, we graph our data and return the figure
    #if(graph):
        
        R, R_sig, capacitance, i_recon, i_corrected = forward_results
        forward_graph = _get_simple_graph(x, R, R_sig, Vfor, Ifor, i_recon, i_corrected)

        R, R_sig, capacitance, i_recon, i_corrected = reverse_results
        reverse_graph = _get_simple_graph(x, R, R_sig, Vrev, Irev, i_recon, i_corrected)

        return forward_graph, reverse_graph
        

    else:'''
    # Concatenate the forward and reverse results together and return in a tuple
    # for easier parallel processing
    # Note, results are (R, R_sig, capacitance, i_recon, i_corrected)
    R = np.concatenate((forward_results[0], reverse_results[0]), axis=0)
    R_sig = np.concatenate((forward_results[1], reverse_results[1]), axis=0)
    capacitance = np.array([forward_results[2], reverse_results[2]])
    i_recon = np.concatenate((forward_results[3], reverse_results[3]), axis=0)
    i_corrected = np.concatenate((forward_results[4], reverse_results[4]), axis=0)
    # Shift i_recon and i_corrected back to correspond to a sine excitation wave
    i_recon = get_unshifted_response(i_recon, shift_index)
    i_corrected = get_unshifted_response(i_corrected, shift_index)

    if(graph):
        # calls publicGetGraph(Ns, pix_ind, shift_index, split_index, x, R, R_sig, V, i_meas, i_recon, i_corrected)
        full_V = get_unshifted_response(full_V, shift_index)
        full_i_meas = get_unshifted_response(full_i_meas, shift_index)
        x = np.concatenate((x, x))
        return publicGetGraph(Ns, pix_ind, shift_index, split_index, x, R, R_sig, full_V, full_i_meas, i_recon, i_corrected)
    else:
        return R, R_sig, capacitance, i_recon, i_corrected

# Helper function because Numba crashes when it isn't supposed to
@jit(nopython=True)
def _logpo_R1_fast(pp, A, V, dV, y, gam, P0, mm):
    out = np.linalg.norm(V*np.exp(-A[:, :-1] @ pp[:-2]) +\
          pp[-2][0] * (dV + pp[-1][0]*V) - y)**2/2/gam/gam + (((pp[:-2]-mm[:-2]).T @ P0) @ (pp[:-2]-mm[:-2]))[0][0]/2
    return out

# Does math stuff and returns a number relevant to some probability distribution.
# Used only in the while loop of run_bayesian_inference() (and once before to initialize)
def _logpo_R1(pp, A, V, dV, y, gam, P0, mm, Rmax, Rmin, Cmax, Cmin):
    if pp[-1] > Rmax or pp[-1] < Rmin:
        return np.inf
    if pp[-2] > Cmax or pp[-2] < Cmin:
        return np.inf
    '''
    out = np.linalg.norm(V*np.exp(np.matmul(-A[:, :-1], pp[:-2])) + \
                         pp[-2][0] * (dV + pp[-1][0]*V) - y)**2/2/gam/gam + \
          np.matmul(np.matmul((pp[:-2]-mm[:-2]).T, P0), pp[:-2]-mm[:-2])/2
    '''
    return _logpo_R1_fast(pp, A, V, dV, y, gam, P0, mm)


def _run_bayesian_inference(V, i_meas, M, dx, x, f, V0, Ns, dvdt, verbose=False):
    '''
    Takes in raw filtered data, parses it down and into forward and reverse sweeps,
    and runs an adaptive metropolis alglrithm on the data. Then calculates the
    projected resistances and variances for both the forward and reverse sweeps, as
    well as the reconstructed current.

    Parameters
    ----------
    V:          numpy.ndtype row vector of dimension 1xN
                the excitation waveform; assumed to be a forward or reverse sweep
    i_meas:     numpy.ndtype row vector of dimension 1xN
                the measured current resulting from the excitation waveform
    f:          int
                the frequency of the excitation waveform (Hz)
    V0:         int
                the amplitude of the excitation waveform (V)
    Ns:         int
                the number of iterations we want the adaptive metropolis to run
    verbose:    boolean
                prints debugging messages if True

    Returns
    -------
    R:              numpy.ndtype column vector
                    the estimated resistances
    R_sig:          numpy.ndtype column vector
                    the standard deviations of R resistances 
    capacitance:    float
                    the capacitance of the setup
    i_recon:        numpy.ndtype column vector
                    the reconstructed current
    i_corrected:    numpy.ndtype column vector
                    the measured current corrected for capacitance
    '''
    # Grab the start time so we can see how long this takes
    if(verbose):
        start_time = time.time()

    # Setup some constants that will be used throughout the code
    nt = 1000;
    nx = 32;

    r_extra = 110
    ff = 1e0

    gam = 0.01
    sigc = 10
    sigma = 1

    tmax = 1/f/2
    t = np.linspace(0, tmax, V.size)
    dt = t[1] - t[0]
    dV = np.diff(V)/dt
    dV = np.append(dV, dV[dV.size-1])
    N = V.size
    #dx = 2*V0/(M-2)
    #x = np.arange(-V0, V0+dx, dx)[np.newaxis].T
    #M = x.size # M may not be the desired value but it will be very close

    # Change V and dV into column vectors for computations
    # Note: V has to be a row vector for np.diff(V) and
    # max(V) to work properly
    dV = dV[np.newaxis].T
    V = V[np.newaxis].T
    i_meas = i_meas[np.newaxis].T

    # Build A : the forward map
    A = np.zeros((N, M + 1))
    for j in range(N):
        # Note: ix will be used to index into arrays, so it is one less
        # than the ix used in the Matlab code
        ix = math.floor((V[j] + V0)/dx) + 1
        ix = min(ix, x.size - 1)
        ix = max(ix, 1)
        A[j, ix] = (V[j] - x[ix-1])/(x[ix] - x[ix-1])
        A[j, ix-1] = (1 - (V[j] - x[ix-1])/(x[ix] - x[ix-1]));
    A[:, M] = (dV + ff*r_extra*V).T 

    # Similar to above, but used to simulate data and invert for E(s|y)
    # for initial condition
    
    A1 = np.zeros((N, M + 1))
    for j in range(N):
        ix = math.floor((V[j] + V0)/dx)+1
        ix = min(ix, x.size - 1)
        ix = max(ix, 1)
        A1[j, ix] = V[j]*(V[j] - x[ix-1])/(x[ix] - x[ix-1])
        A1[j, ix-1] = V[j]*(1 - (V[j] - x[ix-1])/(x[ix] - x[ix-1]))
    A1[:, M] = (dV + ff*r_extra*V).T 

    # A rough guess for the initial condition is a bunch of math stuff
    # This is an approximation of the Laplacian
    # Note: have to do inconvenient things with x because it is a column vector
    Lap = (-np.diag(np.power(x.T[0][:-1], 0), -1) - np.diag(np.power(x.T[0][:-1], 0), 1) 
        + 2*np.diag(np.power(x.T[0], 0), 0))/dx/dx
    Lap[0, 0] = 1/dx/dx
    Lap[-1, -1] = 1/dx/dx

    P0 = np.zeros((M+1, M+1))
    P0[:M, :M] = (1/sigma/sigma)*(np.eye(M) + np.matmul(Lap, Lap))
    P0[M, M] = 1/sigc/sigc

    Sigma = np.linalg.inv(np.matmul(A1.T, A1)/gam/gam + P0)
    m = np.matmul(Sigma, np.matmul(A1.T, i_meas)/gam/gam)

    # Tuning parameters
    Mint = 1000
    Mb = 100
    r = 1.1
    beta = 1
    nacc = 0
    #P = np.zeros((M+2, Ns))
    # Only store a million samples to save space
    num_samples = min(Ns, int(1e6))
    P = np.zeros((M+2, num_samples))

    # Define prior
    SS = np.matmul(spla.sqrtm(Sigma), np.random.randn(M+1, num_samples)) + np.tile(m, (1, num_samples))
    RR = np.concatenate((np.log(1/np.maximum(SS[:M, :], np.full((SS[:M, :]).shape, 
        np.finfo(float).eps))), SS[M, :][np.newaxis]), axis=0)
    mr = 1/num_samples*np.sum(RR, axis=1)[np.newaxis].T
    SP = 1/num_samples*np.matmul(RR - np.tile(mr, (1, num_samples)), (RR - np.tile(mr, (1, num_samples))).T)
    amp = 100
    C0 = amp**2 * SP[:M, :M]
    SR = spla.sqrtm(C0)

    # Initial guess for Sigma from Eq 1.8 in the notes
    S = np.concatenate((np.concatenate((SR, np.zeros((2, M))), axis=0), 
        np.concatenate((np.zeros((M, 2)), amp*np.array([[1e-2, 0], [0, 1e-1]])), axis=0)), axis=1)
    S2 = np.matmul(S, S.T)
    S1 = np.zeros((M+2, 1))
    mm = np.append(mr, r_extra)[np.newaxis].T
    ppp = mm.astype(np.float64)

    # for some reason, this may throw a np.linalg.LinAlgError: Singular Matrix
    # when trying to process the 0th pixel. After exiting this try catch block,
    # the concatinations in process pixel fails.
    try:
        P0 = np.linalg.inv(C0)
    except np.linalg.LinAlgError:
        print("P0 failed to instantiate.")
        # return zero versions of R, R_sig, capacitance, i_recon, i_corrected
        return np.zeros(x.shape), np.zeros(x.shape), 0, np.zeros(i_meas.shape), np.zeros(i_meas.shape)
    

    # Now we are ready to start the active metropolis
    if verbose:
        print("Starting active metropolis...")
        met_start_time = time.time()

    i = 0
    j = 0
    Rmax = 120
    Rmin = 100
    Cmax = 10
    Cmin = 0
    S3 = 0
    logpold = _logpo_R1(ppp, A, V, dV, i_meas, gam, P0, mm, Rmax, Rmin, Cmax, Cmin)

    while i < Ns:
        pppp = ppp + beta*np.matmul(S, np.random.randn(M+2, 1)).astype(np.float64) # using pp also makes gdb bug out
        logpnew = _logpo_R1(pppp, A, V, dV, i_meas, gam, P0, mm, Rmax, Rmin, Cmax, Cmin)
        
        # accept or reject
        # Note: unlike Matlab's rand, which is a uniformly distributed selection from (0, 1),
        # Python's np.random.rand is a uniformly distributed selection from [0, 1), and can
        # therefore be 0 (by a very small probability, but still). Thus, a quick filter must
        # be made to prevent us from trying to evaluate log(0).
        randBoi = np.random.rand()
        while randBoi == 0:
            randBoi = np.random.rand()
        if np.log(randBoi) < logpold - logpnew:
            ppp = pppp
            logpold = logpnew
            nacc += 1 # count accepted proposals

        # stepsize adaptation
        if (i+1) % Mb == 0:
            # estimate acceptance probability, and keep near 0.234
            rat = nacc/Mb

            if rat > 0.3:
                beta = beta*r
            elif rat < 0.1:
                beta = beta/r

            nacc = 0

        # Save only the last num_samples samples to save space
        P[:, i%num_samples] = ppp.T[0]

        # proposal covariance adaptation
        if (i+1) % Mint == 0:
            # Get index values to fit into the smaller matrix
            jMintStart = (j-Mint)%num_samples
            jMintEnd = j%num_samples
            iMintStart = (i-Mint+1)%num_samples
            iMintEnd = (i+1)%num_samples

            if (j+1) == Mint:
                S1lag = np.sum(P[:, :j] * P[:, 1:j+1], axis=1)[np.newaxis].T / j
            else:
                S1lag = (j - Mint)/j*S1lag + np.sum(P[:, jMintStart:jMintEnd] * P[:, jMintStart+1:jMintEnd+1], axis=1)[np.newaxis].T / j

            S3 = S3 + j/(j+1) * np.matmul(P[:, iMintStart:iMintEnd] - S1, (P[:, iMintStart:iMintEnd] - S1).T)
            # Update meani based on Mint batch
            S1 = (j+1-Mint)/(j+1)*S1 + np.sum(P[:, iMintStart:iMintEnd], axis=1)[np.newaxis].T/(j+1)
            # Update Sigma based on Mint batch
            S2 = (j+1-Mint)/(j+1)*S2 + np.matmul(P[:, iMintStart:iMintEnd], P[:, iMintStart:iMintEnd].T)/(j+1)
            #print("P's shape is {}".format(P.shape))
            # Approximate L such that L*L' = Sigma, where the second term is a
            # decaying regularization
            # for some reason this may throw a np.linalg.LinAlgError: Matrix is not positive definite
            # If this fails, just return zeros
            try:
                #S = np.linalg.cholesky(S2 - np.matmul(S1, S1.T) + (1e-3)*np.eye(M+2)/(j+1))
                S = np.linalg.cholesky(S3/(j+1))
            except np.linalg.LinAlgError:
                print("Initial Cholesky failed on iteration {}. Retrying with larger regularization.".format(i+1))
                try:
                    S = np.linalg.cholesky(S2 - np.matmul(S1, S1.T) + (1e-2)*np.eye(M+2)/(j+1))
                except np.linalg.LinAlgError:
                    print("Cholesky failed again. Stopping inference and returning all zeros.")
                    return np.zeros(x.shape), np.zeros(x.shape), 0, np.zeros(i_meas.shape), np.zeros(i_meas.shape)

            if verbose and ((i+1)%1e5 == 0):
                print("i = {}".format(i+1))

        i += 1
        j += 1

    if verbose:
        print("Finished Adaptive Metropolis!\nAdaptive Metropolis took {}.\nTotal time taken so far is {}.".format(time.time() - met_start_time, time.time() - start_time))

    # m is a column vector, and the last element is the capacitance exponentiated
    #capacitance = math.log(m[-1][0])
    capacitance = pppp[-2][0]
    r_extra = pppp[-1][0]

    # Mean and variance of resistance
    meanr = np.matmul(np.exp(P[:M,:]), np.ones((num_samples, 1))) / num_samples
    mom2r = np.matmul(np.exp(P[:M,:]), np.exp(P[:M,:]).T) / num_samples
    varr = mom2r - np.matmul(meanr, meanr.T)

    R = meanr.astype(np.float)
    R_sig = np.sqrt(np.diag(varr[:M, :M]))[np.newaxis].T.astype(np.float)

    # Reconstruction of the current
    i_recon = V * np.matmul(np.exp(np.matmul(-A[:, :M], P[:M, :])), np.ones((num_samples, 1))) / (num_samples) + \
            np.matmul(np.tile(P[M,:], (N, 1))*(np.tile(dV, (1, num_samples)) + P[M+1, :]*np.tile(V, (1, num_samples))), \
                      np.ones((num_samples, 1))) / num_samples

    # Adjusting for capacitance
    point_i_cap = capacitance * dvdt
    point_i_extra = r_extra * 2 * capacitance * V
    i_corrected = i_meas - point_i_cap - point_i_extra

    #breakpoint()

    return R, R_sig, capacitance, i_recon, i_corrected


def _get_simple_graph(x, R, R_sig, V, i_meas, i_recon, i_corrected):
    # Clean up R and R_sig for unsuccessfully predicted resistances
    for i in range(R_sig.size):
        if np.isnan(R_sig[i]) or R_sig[i] > 100:
            R_sig[i] = np.nan
            R[i] = np.nan

    #breakpoint()

    # Create the figure to be returned
    result = plt.figure()

    # Plot the resistance estimation on the left subplot
    plt.subplot(121)
    plt.plot(x, R, "gx-", label="ER")
    plt.plot(x, R+R_sig, "rx:", label="ER+\u03C3_R")
    plt.plot(x, R-R_sig, "rx:", label="ER-\u03C3_R")
    plt.legend()

    # Plot the current data on the right subplot
    plt.subplot(122)
    plt.plot(V, i_meas, "ro", mfc="none", label="i_meas")
    plt.plot(V, i_recon, "gx-", label="i_recon")
    plt.plot(V, i_corrected, "bo", label="i_corrected")
    plt.legend()

    return result


def publicGetGraph(Ns, pix_ind, shift_index, split_index, x, R, R_sig, V, i_meas, i_recon, i_corrected):
    #return _get_simple_graph(x, R, R_sig, V, i_meas, i_recon, i_corrected)
    rLenHalf = R_sig.size//2

    shiftV = get_shifted_response(V, shift_index)
    i_corrected = get_shifted_response(i_corrected, shift_index)

    # Clean up R and R_sig for unsuccessfully predicted resistances
    for i in range(rLenHalf*2):
        if np.isnan(R_sig[i]) or R_sig[i] > 100:
            R_sig[i] = np.nan
            R[i] = np.nan

    #breakpoint()

    # Create the figure to be returned
    result = plt.figure()

    plt.suptitle("Pixel {} after {} iterations".format(pix_ind, Ns))

    # Plot the forward resistance estimation on the left subplot
    plt.subplot(131)
    plt.title("Forward resistance")
    plt.plot(x[:rLenHalf], R[:rLenHalf], "gx-", label="ER")
    plt.plot(x[:rLenHalf], R[:rLenHalf]+R_sig[:rLenHalf], "rx:", label="ER+\u03C3_R")
    plt.plot(x[:rLenHalf], R[:rLenHalf]-R_sig[:rLenHalf], "rx:", label="ER-\u03C3_R")
    plt.legend()

    # Plot the reverse resistance estimation on the middle subplot
    plt.subplot(132)
    plt.title("Reverse resistance")
    plt.plot(x[rLenHalf:], R[rLenHalf:], "gx-", label="ER")
    plt.plot(x[rLenHalf:], R[rLenHalf:]+R_sig[rLenHalf:], "rx:", label="ER+\u03C3_R")
    plt.plot(x[rLenHalf:], R[rLenHalf:]-R_sig[rLenHalf:], "rx:", label="ER-\u03C3_R")
    plt.legend()

    # Plot the current data on the right subplot
    plt.subplot(133)
    plt.title("Current")
    plt.plot(V, i_meas, "ro", mfc="none", label="i_meas")
    plt.plot(V, i_recon, "gx-", label="i_recon")
    plt.plot(shiftV[:split_index], i_corrected[:split_index], "bx", label="forward i_corrected")
    plt.plot(shiftV[split_index:], i_corrected[split_index:], "yx", label="reverse i_corrected")
    plt.legend()

    return result

