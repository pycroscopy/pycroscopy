# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 21:28:05 2017

@author: Enrique Alejandro

Description: this library contains functions for postprocessing of results of AFM simulations
"""

import numpy as np

def E_diss(z, Fts, dt, fo1):
    """This function calculates the tip-sample dissipation per oscillating period.    
    
    Parameters
    ----------
    z : numpy.ndarray 
        tip deflection
    Fts : numpy.ndarray 
        tip-sample interacting force
    dt : float
        simulation timestep
    fo1 : float
        eigenmode resonance frequency
    
    Returns
    -------    
    Ediss/number_of_periods : float
        total dissipated energy per oscillating period       
    """
    Ediss = 0.0
    for i in range(1,len(z)-1):
        Ediss -= Fts[i]*(z[i+1]-z[i-1])/2.0   #based on integral of Fts*dz/dt*dt, dz/dt=(z[i+1]-z[i-1])/(2.0*dt) Central difference approx
    total_time = dt*len(z)
    period1 = 1.0/ fo1
    number_of_periods = total_time/period1 
    return Ediss/number_of_periods

def V_ts(z, Fts, dt):
    """This function calculates the virial of the interaction.
    
    A more detailed description of this quantity is given in: 
    San Paulo, Alvaro, and Ricardo García. Phys Rev B 64.19 (2001): 193411.
    
    Parameters
    ----------
    z : numpy.ndarray
        tip deflection
    Fts : numpy.ndarray
        tip-sample interacting force
    dt : float
        simulation timestep
    
    Returns
    -------    
    Vts/(dt*len(z)) : float
        virial of the tip-sample interaction
    """
    Vts = 0.0
    for i in range(len(z)):
        Vts = Vts + Fts[i]*z[i]*dt
    return Vts/(dt*len(z))     #virial is 1/T*S(Fts*z*dt) from 0 to T, being T total experimental time

def av_dt(array):
    """this function returns the average of the timesteps in a time array.
    
    Parameters
    ----------
    array : numpy.ndarray
        generally unequally spaced time-array
    
    Returns
    ------- 
    dt: float
        averaged timestep of the unequally space time array
    """
    i = 0
    k = 0.0
    for i in range(np.size(array)-1):
        k = k + (array[i+1]-array[i])
        dt = k/(np.size(array)-1)
    return dt

def Amp_Phase(t, f_t, freq):
    """this function calculates amplitude and phase using the in-phase and in-quadrature integrals for a given frequency.
    
    Parameters
    ----------
    t : numpy.ndarray
        time array of the simulation
    f_t: numpy.ndarray
        signal in time whose amplitude and phase at certain frequency is extracted
    freq : float
        distinct frequency at which the amplitude and phase will be calculated
        
    Returns
    ------- 
    Amp : float
        amplitude of the signal related to freq
    Phase : float
        Phase pf the signal related to freq
    """    
    if t[0] > 0.0:
        t-= t[0]
    dt = av_dt(t)
    I = 0.0
    K = 0.0
    for i in range(np.size(f_t)):
        I = I + f_t[i]*np.cos(2.0*np.pi*freq*t[i])*dt
        K = K + f_t[i]*np.sin(2.0*np.pi*freq*t[i])*dt
    Amp = 1.0/(t[np.size(t)-1])*np.sqrt(I**2+K**2) *2.0
    Phase = np.arctan(K/I)*180.0/np.pi
    if Phase < 0.0:
        Phase = Phase + 180.0
    return Amp, Phase

def Ediss_obs(k, Q, A_free, A, Phase):
    """Dissipated energy calculated from the dynamic AFM observables. 
    
    Equation details can be seen in: J Tamayo, R Garcı́a Applied Physics Letters 73 (20), 2926-2928
    
    Parameters
    ----------
    k : float
        eigenmode's stiffness
    Q : float
        eigenmode's quality factor
    A_free : float
        free oscillating amplitude (oscillating amplitude in the absence of tip-sample interaction)
    A : float
        tapping amplitude (oscillating amplitude in the presence of tip-sample interaction)
    
    Returns
    ------- 
    Ediss: float
        dissipated energy per oscillating period (calculated from AFM observables)
    """
    Ediss = (np.pi*k*A**2/Q)*( (A_free/A)*np.sin(Phase*np.pi/180.0) - 1.0 )
    return Ediss

def virial_obs(k, Q, A_free, A, Phase):
    """Virial of the interaction calculated from the dynamic AFM observables.
    
    Details of the equation in: San Paulo, Alvaro, and Ricardo García. Phys Rev B 64.19 (2001): 193411.
    
    Parameters
    ----------
    k : float
        eigenmode's stiffness
    Q : float
        eigenmode's quality factor
    A_free : float
        free oscillating amplitude (oscillating amplitude in the absence of tip-sample interaction)
    A : float
        tapping amplitude (oscillating amplitude in the presence of tip-sample interaction)
    
    Returns
    ------- 
    Vts: float
        virial of the interaction (calculated from AFM observables)
    """
    Vts = -(k*A*A_free)/(2.0*Q)*np.cos(Phase*np.pi/180.0)
    return Vts

def derivative_cd(f_t, t):
    """this function calculates the derivative of a given array using central difference scheme.
    
    Parameters
    ----------
    t : numpy.ndarray
        time trace
    f_t: numpy.ndarray
        function trace whose 1st derivative is to be numerically calculated using the central difference scheme
        
    Returns
    ------- 
    f_prime : numpy.ndarray
        first derivative of the f_t trace
    """
    f_prime = np.zeros(np.size(f_t))
    for i in range(np.size(f_t)):  #calculation of derivative using central difference scheme
        if i == 0:
            f_prime[i] = (f_t[1]-f_t[0])/(t[1]-t[0])
        else:
            if i == np.size(f_t)-1:
                f_prime[np.size(f_t)-1] = (f_t[np.size(f_t)-1]-f_t[np.size(f_t)-2])/(t[np.size(f_t)-1]-t[np.size(f_t)-2])
            else:
                f_prime[i] = (f_t[i+1]-f_t[i-1])/(t[i+1]-t[i-1])
    return f_prime

