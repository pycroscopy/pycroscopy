# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 21:28:05 2017

@author: Enrique Alejandro

Description: this library contains functions for postprocessing of results of AFM simulation
"""
from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np


def e_diss(defl, f_ts, dt, fo1):
    """This function calculates the tip-sample dissipation per oscillating period.    
    
    Parameters
    ----------
    defl : numpy.ndarray
        tip deflection
    f_ts : numpy.ndarray
        tip-sample interacting force
    dt : float
        simulation timestep
    fo1 : float
        eigenmode resonance frequency
    
    Returns
    -------    
    energy_diss/number_of_periods : float
        total dissipated energy per oscillating period       
    """
    energy_diss = 0.0
    for i in range(1, len(defl) - 1):
        # based on integral of f_ts*dz/dt*dt, dz/dt=(defl[i+1]-defl[i-1])/(2.0*dt) Central difference approx
        energy_diss -= f_ts[i] * (defl[i + 1] - defl[i - 1]) / 2.0
    total_time = dt*len(defl)
    period1 = 1.0 / fo1
    number_of_periods = total_time/period1 
    return energy_diss/number_of_periods


def v_ts(defl, f_ts, dt):
    """This function calculates the virial of the interaction.
    
    A more detailed description of this quantity is given in: 
    San Paulo, Alvaro, and Ricardo García. Phys Rev B 64.19 (2001): 193411.
    
    Parameters
    ----------
    defl : numpy.ndarray
        tip deflection
    f_ts : numpy.ndarray
        tip-sample interacting force
    dt : float
        simulation timestep
    
    Returns
    -------    
    virial_tip_samp/(dt*len(defl)) : float
        virial of the tip-sample interaction
    """
    virial_tip_samp = 0.0
    for i in range(len(defl)):
        virial_tip_samp = virial_tip_samp + f_ts[i] * defl[i] * dt

    # virial is 1/T*S(f_ts*defl*dt) from 0 to T, being T total experimental time
    return virial_tip_samp/(dt * len(defl))


def av_dt(array):
    """this function returns the average of the time steps in a time array.
    
    Parameters
    ----------
    array : numpy.ndarray
        generally unequally spaced time-array
    
    Returns
    ------- 
    dt: float
        averaged timestep of the unequally space time array
    """
    k = 0.0
    dt = 0
    for ind in range(np.size(array)-1):
        k = k + (array[ind+1]-array[ind])
        dt = k/(np.size(array)-1)
    return dt


def amp_phase(time_vec, f_t, freq):
    """this function calculates amplitude and phase using the in-phase and in-quadrature integrals for a given frequency
    
    Parameters
    ----------
    time_vec : numpy.ndarray
        time array of the simulation
    f_t: numpy.ndarray
        signal in time whose amplitude and phase at certain frequency is extracted
    freq : float
        distinct frequency at which the amplitude and phase will be calculated
        
    Returns
    ------- 
    amp : float
        amplitude of the signal related to freq
    Phase : float
        Phase pf the signal related to freq
    """    
    if time_vec[0] > 0.0:
        time_vec -= time_vec[0]
    dt = av_dt(time_vec)
    i_val = 0.0
    k_val = 0.0
    for ind in range(np.size(f_t)):
        i_val = i_val + f_t[ind] * np.cos(2.0 * np.pi * freq * time_vec[ind]) * dt
        k_val = k_val + f_t[ind] * np.sin(2.0 * np.pi * freq * time_vec[ind]) * dt
    amp = 1.0 / (time_vec[np.size(time_vec) - 1]) * np.sqrt(i_val ** 2 + k_val ** 2) * 2.0
    phase = np.arctan(k_val/i_val)*180.0/np.pi
    if phase < 0.0:
        phase = phase + 180.0
    return amp, phase


def e_diss_obs(stiffness, qual_fac, amp_free, amp_ts, phase):
    """Dissipated energy calculated from the dynamic AFM observables. 
    
    Equation details can be seen in: J Tamayo, R Garcı́a Applied Physics Letters 73 (20), 2926-2928
    
    Parameters
    ----------
    stiffness : float
        eigenmode's stiffness
    qual_fac : float
        eigenmode's quality factor
    amp_free : float
        free oscillating amplitude (oscillating amplitude in the absence of tip-sample interaction)
    amp_ts : float
        tapping amplitude (oscillating amplitude in the presence of tip-sample interaction)
    phase : float
        phase
    
    Returns
    ------- 
    e_diss: float
        dissipated energy per oscillating period (calculated from AFM observables)
    """
    e_diss = (np.pi * stiffness * amp_ts ** 2 / qual_fac) * ((amp_free / amp_ts) * np.sin(phase * np.pi / 180.0) - 1.0)
    return e_diss


def virial_obs(stiffness, qual_fac, amp_free, amp_ts, phase):
    """Virial of the interaction calculated from the dynamic AFM observables.
    
    Details of the equation in: San Paulo, Alvaro, and Ricardo García. Phys Rev B 64.19 (2001): 193411.
    
    Parameters
    ----------
    stiffness : float
        eigenmode's stiffness
    qual_fac : float
        eigenmode's quality factor
    amp_free : float
        free oscillating amplitude (oscillating amplitude in the absence of tip-sample interaction)
    amp_ts : float
        tapping amplitude (oscillating amplitude in the presence of tip-sample interaction)
    phase : float
        phase
    
    Returns
    ------- 
    v_ts: float
        virial of the interaction (calculated from AFM observables)
    """
    v_ts = -(stiffness * amp_ts * amp_free) / (2.0 * qual_fac) * np.cos(phase * np.pi / 180.0)
    return v_ts


def derivative_cd(f_t, time_vec):
    """this function calculates the derivative of a given array using central difference scheme.
    
    Parameters
    ----------
    f_t: numpy.ndarray
        function trace whose 1st derivative is to be numerically calculated using the central difference scheme
    time_vec : numpy.ndarray
        time trace
        
    Returns
    ------- 
    f_prime : numpy.ndarray
        first derivative of the f_t trace
    """
    f_prime = np.zeros(np.size(f_t))
    for i in range(np.size(f_t)):  # calculation of derivative using central difference scheme
        if i == 0:
            f_prime[i] = (f_t[1]-f_t[0])/(time_vec[1] - time_vec[0])
        else:
            if i == np.size(f_t)-1:
                f_prime[np.size(f_t)-1] = (f_t[np.size(f_t)-1]-f_t[np.size(f_t)-2]) / \
                                          (time_vec[np.size(f_t) - 1] - time_vec[np.size(f_t) - 2])
            else:
                f_prime[i] = (f_t[i+1]-f_t[i-1])/(time_vec[i + 1] - time_vec[i - 1])
    return f_prime
