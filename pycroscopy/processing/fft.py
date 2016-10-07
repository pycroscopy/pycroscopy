# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:42:41 2015

@author: Suhas Somnath
"""
###############################################################################

import numpy as np  # for all array, data operations
import matplotlib.pyplot as plt  # for all plots
from scipy.special import erf  
from warnings import warn


def getNoiseFloor(fft_data,tolerance):
    """
    Paramters
    ---------
    fft_data : 1D or 2D complex numpy array
        Signal in frequency space (ie - after FFT shifting) arranged as (channel or repetition, signal)
    tolerance : unsigned float
        Tolerance to noise. A smaller value gets rid of more noise.
        
    Returns
    -------
    noise_floor : 1D real numpy array 
        One value per channel / repetition
    """
    
    fft_data = np.atleast_2d(fft_data)
    # Noise calculated on the second axis
    
    noise_floor = np.zeros(fft_data.shape[0])

    for chan in xrange(fft_data.shape[0]):

        amp=np.abs(fft_data[chan,:])
        num_pts = amp.size
        temp=np.sqrt(np.sum(amp**2)/(2*num_pts))
        threshold=np.sqrt((2*temp**2)*(-np.log(tolerance)))
        
        bdiff=1        
        B = list()
        B.append(temp)
                
        while (bdiff>10**-2) and len(B)<50:
            amp[amp>threshold]=0
            temp=np.sqrt(sum(amp**2)/(2*num_pts)) 
            B.append(temp)
            bdiff=abs(B[-1]-B[-2])    
            threshold=np.sqrt((2*temp**2)*(-np.log(tolerance)))    

        noise_floor[chan] = threshold
        
    return noise_floor

###############################################################################

def downSample(F_vec, freq_ratio):
    """
    Downsamples the provided data vector
    
    Parameters:
    -----------
    F_vec : 1D complex numpy array
        Waveform that is already FFT shifted
    freq_ratio : float
        new sampling rate / old sampling rate (less than 1)
    
    Returns:
    --------
    F_vec : 1D numpy array 
        downsampled waveform
    """
    if freq_ratio >= 1:
        print('Error at downSample: New sampling rate > old sampling rate')
        return F_vec
        
    vec_len = len(F_vec)
    ind = np.round(float(vec_len)*(0.5*float(freq_ratio)))
    F_vec = F_vec[max(0.5*vec_len-ind,0):min(0.5*vec_len+ind,vec_len)]
    F_vec = F_vec * freq_ratio * 2;
    
    return np.fft.ifft(np.fft.ifftshift(F_vec))


############################################################################### 

def noiseBandFilter(num_pts,samp_rate,freqs,freq_widths, show_plots=False):
    """
    Builds a filter that removes specified noise frequencies
    
    Parameters:
    ---------------------    
    num_pts : unsigned int
        Number of points in the FFT signal
    samp_rate : unsigned int
        sampling rate    
    freqs : 1D array or list
        Target frequencies as unsigned ints    
    freq_widths : 1D array or list
        Width around the target frequency that should be set to 0\n    
    
    Note: sampRate, freqs, freq_widths have same units - eg MHz
    
    Returns:
    ----------
    noise_filter : 1D numpy array
        Array of ones set to 0 at noise bands
    """
    num_pts = abs(int(num_pts))
       
    # Making code a little more robust with handling different inputs:
    samp_rate = float(samp_rate);
    freqs = np.array(freqs);
    freq_widths = np.array(freq_widths);
    if freqs.ndim != freq_widths.ndim:
        warn('Error in noiseBandFilter: dimensionality of frequencies and frequency widths do not match!');
        return None
    if freqs.shape != freq_widths.shape:
        warn('Error in noiseBandFilter: shape of frequencies and frequency widths do not match!');
        return None
        
    cent = int(round(0.5*num_pts))
    
    noise_filter = np.ones(num_pts,dtype=np.int16)
    
    if show_plots:
        w_vec = np.arange(-0.5*samp_rate, 0.5*samp_rate, samp_rate/num_pts);
        fig, ax = plt.subplots(2,1)
        ax[0].plot(w_vec,noise_filter)
        ax[0].set_yscale('log') 
        ax[0].axis('tight');
        ax[0].set_xlabel('Freq')
        ax[0].set_title('Before clean up')

    # Setting noise freq bands to 0
    for cur_freq, d_freq in zip(freqs,freq_widths): 
        ind = int(round(num_pts*(cur_freq/samp_rate)))
        sz = int(round(cent*d_freq/samp_rate))
        noise_filter[cent-ind-sz:cent-ind+sz+1] = 0
        noise_filter[cent+ind-sz:cent+ind+sz+1] = 0

    if show_plots:
        ax[1].plot(w_vec,noise_filter)
        ax[1].set_yscale('log')
        ax[1].axis('tight')
        ax[1].set_xlabel('Freq')
        ax[1].set_title('After clean up')
        plt.show()
    
    return noise_filter
    
###############################################################################
    
def makeLPF(num_pts,samp_rate, f_cutoff, roll_off=0.05):
    """
    Builds a low pass filter
    
    Paramters:
    -----------
    num_pts : unsigned int
        Points in the FFT. Assuming Signal in frequency space (ie - after FFT shifting)
    samp_rate : unsigned integer
        Sampling rate
    f_cutoff : unsigned integer
        Cutoff frequency for filter
    roll_off : 0 < float < 1
        Frequency band over which the filter rolls off. rol off = 0.05 on a 
        100 kHz low pass filter -> roll off from 95 kHz (1) to 100 kHz (0)
        
    Returns:
    -----------
    LPF : 1D numpy array describing the low pass filter
    """
    
    num_pts = abs(int(num_pts))
    
    cent = int(round(0.5*num_pts))
    
    if f_cutoff >= 0.5 * samp_rate:
        print('Error in LPFClip --> LPF too high! Skipping');
        return

    #BW = 0.1; %MHz - Nothing beyond BW.
    roll_off *= f_cutoff  #MHz

    sz = int(np.round(num_pts*(roll_off/samp_rate)))
    ind = int(np.round(num_pts*(f_cutoff/samp_rate)))

    LPF = np.zeros(num_pts, dtype=np.float32)
    
    extent=5
    t2 = np.linspace(-extent/2,extent/2,num=sz)
    smoothing = 0.5*(1+erf(t2))
    
    LPF[cent-ind:cent-ind+sz] = smoothing
    LPF[cent-ind+sz:cent+ind-sz+1] = 1
    LPF[cent+ind-sz+1:cent+ind+1] = 1-smoothing
        
    return LPF # return the filter itself so that it may be repeatedly used outside
    
###############################################################################
    
def harmonicsPassFilter(num_pts,samp_rate,first_freq,band_width,num_harm, doPlots=False):
    """
    Builds a filter that only keeps N harmonics
    
    Parameters:
    -------------
    num_pts : unsigned int
        Number of points in the FFt signal
    samp_rate : unsigned int
        Sampling rate
    first_freq : unsigned int
        Frequency of the first harmonic
    band_width : unsigned int
        Frequency band around each harmonic that needs to be preserved
    num_harm : unsigned int
        Number of harmonics to preserve
    do_plots : Boolean (optional)
        Whether or not to generate plots. Not necessary after debugging

    Note that the frequency values must all have the same units
    
    Returns:
    ---------
    harm_filter : 1D numpy array
        0s where the signal is to be rejected and 1s at harmonics
        
    """
    
    num_pts = abs(int(num_pts))
    
    harm_filter = np.ones(num_pts, dtype=np.int16)
    
    cent = int(round(0.5*num_pts))
        
    if doPlots:    
        print('OnlyKeepHarmonics: samp_rate = %2.1e Hz, first harmonic = %3.2f Hz, %d harmonics w/- %3.2f Hz bands\n' %(samp_rate,first_freq,num_harm,band_width))     
        w_vec = np.arange(-samp_rate/2,samp_rate/2,samp_rate/num_pts)
        fig, ax = plt.subplots(figsize=(5,5)); ax.plot(w_vec,harm_filter); ax.set_title('Raw')

    sz = int(round(cent*band_width/samp_rate))

    # First harmonic
    ind = int(round(num_pts*(first_freq/samp_rate)))
    if ind >= num_pts:
        return None

    harm_filter[max(cent-ind+sz+1,0):min(num_pts,cent+ind-sz)] = 0
    
    if doPlots:
        fig2, ax2 = plt.subplots(figsize=(5,5)); ax2.plot(w_vec,harm_filter); ax2.set_title('Step 1')

    # Last harmonic
    ind = int(round(num_pts*(num_harm*first_freq/samp_rate)))
    harm_filter[:cent-ind-sz] = 0
    harm_filter[cent+ind+sz+1:] = 0
    
    if doPlots:
    	fig3, ax3 = plt.subplots(figsize=(5,5)); ax3.plot(w_vec,harm_filter); ax3.set_title('Step 2')

    if num_harm == 1:
        return harm_filter

    for harm_ind in xrange(1,num_harm):
        ind = int(round(num_pts*(harm_ind*first_freq/samp_rate)))
        ind2 = int(round(num_pts*((harm_ind+1)*first_freq/samp_rate)))
        harm_filter[cent-ind2+sz+1:cent-ind-sz] = 0
        harm_filter[cent+ind+sz+1:cent+ind2-sz] = 0
        if doPlots:
            fig4, ax4 = plt.subplots(figsize=(5,5)); ax4.plot(w_vec,harm_filter); ax4.set_title('Step %d' %(harm_ind+2))
    
    return harm_filter
    
    
###############################################################################
    
# def removeNoiseHarmonics(F_AI_vec,samp_rate,noise_combs):
#     """
#     Removes specified noise frequencies from the signal
#     
#     Parameters:
#     ---------------------
#     * F_AI_vec -- matrix (chan x pts) already FFT + FFT shifted
#     
#     * sampRate -- sampling rate
#     
#     * freqs -- 1D array of target frequencies
#     
#     * freqWidths -- 1D array of frequency windows that correspond to freqs that should be set to 0
#     
#     * Note: sampRate, freqs, freqWidths have same units - eg MHz
#     """
#     numpts = size(F_AI_vec,2);
#     freqs = []; freqWidths = [];
#     for k1 = 1:length(noiseCombs):
#         stFreq = noiseCombs(k1).first_harm_freq;
#         nBands = noiseCombs(k1).num_harmonics;
#         if nBands < 0
#             nBands = 
#         end
#     end
#     F_AI_vec = removeNoiseFreqs(F_AI_vec,sampRate,freqs,freqWidths);    