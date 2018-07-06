# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:20:30 2018

@author: Enrique Alejandro

Description:  This library contains useful rheology based functions, for example for the interconversion
between generalized Maxwell and Voigt models, calculator of operator coefficients from Maxwell or Voigt 
parameters.
"""

import numpy as np
import sys
sys.path.append('d:\github\pycroscopy')
from pycroscopy.simulation.afm_calculations import av_dt


  
def j_storage(omega, Jg, J, tau):
    """this function gives an array of storage compliance on radian frequency
    
    Parameters:
    ---------- 
    omega : numpy.ndarray
        array of frequencies at which the loss compliance will be calculated
    Jg : float
        glassy compliance of the material
    J : numpy.ndarray or float
        array with compliances of the springs in the generalized voigt model, or single compliance in case of SLS model
    tau: numpy.ndarray or float
        array with retardation times in the generalized voigt model, or single tau in case if SLS model
    
    Returns:
    ---------- 
    J_prime : numpy.ndarray
        calculated storage moduli corresponding to the passed frequencies (omega)    
    """
    J_prime = np.zeros(len(omega))
    for i in range(len(omega)):
        if np.size(J) > 1:
            J_prime[i] = Jg + sum( J[:]/(1.0 + (  pow(omega[i],2)*pow(tau[:],2) ) ) )
        else: #the material is the standard linear solid (only one retardation time present)
            J_prime[i] = Jg + ( J/(1.0 + (  pow(omega[i],2)*pow(tau,2) ) ) )
            
    return J_prime

def j_loss(omega, Jg, J, tau, phi = 0.0):
    """this function returns an array of loss compliance on radian frequency
    
    Parameters:
    ---------- 
    omega : numpy.ndarray
        array of frequencies at which the loss compliance will be calculated
    Jg : float
        glassy compliance of the material
    J : numpy.ndarray or float
        array with compliances of the springs in the generalized voigt model, or single compliance in case of SLS model
    tau: numpy.ndarray or float
        array with retardation times in the generalized voigt model, or single tau in case if SLS model
    phi : float, optional
        steady state fluidity
    
    Returns:
    ---------- 
    J_biprime : numpy.ndarray
        calculated loss moduli corresponding to the passed frequencies (omega)    
    """
    J_biprime = np.zeros(len(omega))
    for i in range(len(omega)):
        if np.size(J)>1:
            J_biprime[i] = sum( J[:]*omega[i]*tau[:]/(1.0 + (pow(omega[i],2)*pow(tau[:],2)) ) ) + phi/omega[i]
        else:
            J_biprime[i] = ( J*omega[i]*tau/(1.0 + (pow(omega[i],2)*pow(tau,2)) ) ) + phi/omega[i]
    return J_biprime

def theta_v(omega, Jg, J, tau, phi = 0.0):    
    """this function returns an array of loss angle on radian frequency
    
    Parameters:
    ---------- 
    omega : numpy.ndarray
        array of frequencies at which the loss compliance will be calculated
    Jg : float
        glassy compliance of the material
    J : numpy.ndarray or float
        array with compliances of the springs in the generalized voigt model, or single compliance in case of SLS model
    tau: numpy.ndarray or float
        array with retardation times in the generalized voigt model, or single tau in case if SLS model
    phi : float, optional
        steady state fluidity
    
    Returns:
    ---------- 
    theta : numpy.ndarray
        calculated loss angle corresponding to the passed frequencies (omega)    
    """
    Jloss = j_loss(omega, Jg, J, tau, phi)
    Jstorage =  j_storage(omega, Jg, J, tau)
    theta = np.arctan(Jloss/Jstorage)*180/np.pi
    return theta

def g_loss(omega, G, tau, Ge = 0.0):
    """this function returns the value of G_loss for either a point value or an array of omega
    
    Parameters:
    ---------- 
    omega : numpy.ndarray
        array of frequencies at which the loss moduli will be calculated
    G : numpy.ndarray or float
        array with moduli of the springs in the generalized maxwell model, or single modulus in case of SLS model
    tau: numpy.ndarray or float
        array with relaxation times in the generalized maxwell model, or single tau in case if SLS model
    Ge : float, optional
        equilibrium modulus
    
    Returns:
    ---------- 
    G_biprime : numpy.ndarray
        calculated loss moduli corresponding to the passed frequencies (omega)    
    """
    if np.size(omega) == 1:  #calculation of point value of G_loss
        G_biprime = 0.0
        if np.size(G) >1: #the model has more than one arm
            G_biprime = sum(  G[:]*omega*tau[:]/( 1.0+pow(omega,2)*pow(tau[:],2) )  )
        else:  #the modela is the SLS
            G_biprime = (  G*omega*tau/( 1.0+pow(omega,2)*pow(tau,2) )  )
    else: #calculation of G_loss for an array of omega
        G_biprime = np.zeros(np.size(omega))
        if np.size(G) > 1: #the model has more than one arm
            for j in range(np.size(omega)):
                G_biprime[j] = sum(  G[:]*omega[j]*tau[:]/( 1.0+pow(omega[j],2)*pow(tau[:],2) )  )
        else: # the model is the SLS
            for j in range(np.size(omega)):
                G_biprime[j] = (  G*omega[j]*tau/( 1.0+pow(omega[j],2)*pow(tau,2) )  )              
    return G_biprime

def g_storage(omega, G, tau, Ge = 0.0):
    """this function returns the value of G_store for either a point value or an array of omega
    
    Parameters:
    ---------- 
    omega : numpy.ndarray
        array of frequencies at which the loss moduli will be calculated
    G : numpy.ndarray or float
        array with moduli of the springs in the generalized maxwell model, or single modulus in case of SLS model
    tau: numpy.ndarray or float
        array with relaxation times in the generalized maxwell model, or single tau in case if SLS model
    Ge : float, optional
        equilibrium modulus
    
    Returns:
    ---------- 
    G_prime : numpy.ndarray
        calculated storage moduli corresponding to the passed frequencies (omega)    
    """
    if np.size(omega) == 1:  #calculation of point value of G_loss
        G_prime = 0.0
        if np.size(G) >1: #the model has more than one arm
            Gg = Ge+sum(G[:])
            G_prime = Gg - sum(  G[:]/( 1.0+pow(omega,2)*pow(tau[:],2) )  )
        else:  #the modela is the SLS
            Gg = Ge+G
            G_prime = Gg - (  G/( 1.0+pow(omega,2)*pow(tau,2) )  )
    else: #calculation of G_loss for an array of omega
        G_prime = np.zeros(np.size(omega))
        if np.size(G) > 1: #the model has more than one arm
            Gg = Ge+sum(G[:])
            for j in range(np.size(omega)):
                G_prime[j] = Gg - sum(  G[:]/( 1.0+pow(omega[j],2)*pow(tau[:],2) )  )
        else: # the model is the SLS
            for j in range(np.size(omega)):
                Gg = Ge+G
                G_prime[j] = Gg - (  G/( 1.0+pow(omega[j],2)*pow(tau,2) )  )             
    return G_prime    
    
def theta_g(omega, G, tau, Ge = 0):
    """this function returns the loss angle from Generalized Maxwell Prony Coefficients
    
    Parameters:
    ---------- 
    omega : numpy.ndarray
        array of frequencies at which the loss moduli will be calculated
    G : numpy.ndarray or float
        array with moduli of the springs in the generalized maxwell model, or single modulus in case of SLS model
    tau: numpy.ndarray or float
        array with relaxation times in the generalized maxwell model, or single tau in case if SLS model
    Ge : float, optional
        equilibrium modulus
        
    Returns:
    ---------- 
    theta : numpy.ndarray
        calculated loss angle corresponding to the passed frequencies (omega)    
    """
    Gloss = g_loss(omega, G, tau, Ge)
    Gstorage = g_storage(omega, G, tau, Ge)
    theta = np.arctan(Gloss/Gstorage)*180.0/np.pi
    return theta       


def chi_th(t, Jg, J, tau, phi = 0):
    """this function gives the strain response to a unit slope stress (the time varying fluidity)
    
    Parameters:
    ---------- 
    t : numpy.ndarray
        time trace
    Jg : float
        glassy compliance of the material
    J : numpy.ndarray or float
        array with compliances of the springs in the generalized voigt model, or single compliance in case of SLS model
    tau: numpy.ndarray or float
        array with retardation times in the generalized voigt model, or single tau in case if SLS model
    phi : float, optional
        steady state fluidity
    
    Returns:
    ---------- 
    chi : numpy.ndarray
        calculated time varying fluidity   
    """
    if (len(J)) > 1:
        Je = sum(J[:])+Jg
    else:
        Je = J+Jg
    chi = np.zeros(len(t))
    for i in range (len(t)):
        if (len(J)) >1 :
            chi[i] = Je*t[i] + sum(J[:]*tau[:]*(np.exp(-t[i]/tau[:])-1.0)) + 1/2.0*phi*pow(t[i],2)
        else:
            chi[i] = Je*t[i] + (J*tau*(np.exp(-t[i]/tau)-1.0)) + 1/2.0*phi*pow(t[i],2)
    return chi

def j_t(t, Jg, J, tau, phi=0):
    """this function returns the compliance in time t, for a model with given set of parameters
    
    Parameters:
    ---------- 
    t : numpy.ndarray
        time trace
    Jg : float
        glassy compliance of the material
    J : numpy.ndarray or float
        array with compliances of the springs in the generalized voigt model, or single compliance in case of SLS model
    tau: numpy.ndarray or float
        array with retardation times in the generalized voigt model, or single tau in case if SLS model
    phi : float, optional
        steady state fluidity
    
    Returns:
    ---------- 
    comp : numpy.ndarray
        calculated theoretical creep compliance   
    """
    if (np.size(J)) > 1:
        Je = sum(J[:])+Jg
    else:
        Je = J+Jg
    comp = np.zeros(len(t))
    for i in range (len(t)):
        if np.size(J) >1:
            comp[i] = Je - sum(J[:]*np.exp(-t[i]/tau[:])) + phi*t[i]
        else: #the model is the SLS
            comp[i] = Je - (J*np.exp(-t[i]/tau)) + phi*t[i]
    return comp

def g_t(t, G, tau, Ge = 0.0):
    """this function returns the relaxation modulus in time
    
    Parameters:
    ---------- 
    t : numpy.ndarray
        time trace
    G : numpy.ndarray or float
        array with moduli of the springs in the generalized maxwell model, or single modulus in case of SLS model
    tau: numpy.ndarray or float
        array with relaxation times in the generalized maxwell model, or single tau in case if SLS model
    Ge : float, optional
        equilibrium modulus
        
    Returns:
    ---------- 
    G_rel : numpy.ndarray
        calculated relaxation modulus    
    """
    G_rel = np.zeros(np.size(t))
    if np.size(G) == 1:  #the model is the SLS
        for i in range(np.size(t)):
            G_rel[i] = Ge + G*np.exp(-t[i]/tau)
    else: #the model has more than one arm
        for i in range(np.size(t)):
            G_rel[i] = Ge + sum(G[:]*np.exp(-t[i]/tau[:]))
    return G_rel

def u_t(t, J, tau, phi=0):
    """this function gives the response of a unit strain impulse
    
    It does not contain the term with the delta function: $J_g \delta (t)$, which has to be analitycally added
    
    Parameters:
    ---------- 
    t : numpy.ndarray
        time trace
    Jg : float
        glassy compliance of the material
    J : numpy.ndarray or float
        array with compliances of the springs in the generalized voigt model, or single compliance in case of SLS model
    tau: numpy.ndarray or float
        array with retardation times in the generalized voigt model, or single tau in case if SLS model
    phi : float, optional
        steady state fluidity
    
    Returns:
    ---------- 
    U : numpy.ndarray
        calculated theoretical unit strain impulse: $U(t) - J_g \delta (t)$
    """
    U = np.zeros(len(t))
    for i in range(len(t)):
        if np.size(J) > 1:
            U[i] = sum(J[:]/tau[:]*np.exp(-t[i]/tau[:])) + phi
        else: #the model is the SLS
            U[i] = J/tau*np.exp(-t[i]/tau) + phi
            
    return U

def conv_uf(t, F, Jg, J, tau, phi=0):
    """this function convolves force and the retardance U(t)
    
    Parameters:
    ---------- 
    t : numpy.ndarray
        time trace
    Jg : float
        glassy compliance of the material
    J : numpy.ndarray or float
        array with compliances of the springs in the generalized voigt model, or single compliance in case of SLS model
    tau: numpy.ndarray or float
        array with retardation times in the generalized voigt model, or single tau in case if SLS model
    phi : float, optional
        steady state fluidity
    
    Returns:
    ---------- 
    U : numpy.ndarray
        calculated theoretical unit strain impulse: $U(t) - J_g \delta (t)$
    """
    dt = av_dt(t)
    U = u_t(t, J, tau, phi)
    conv = np.convolve(U, F, mode='full')*dt
    conv = conv[range(len(F))] + Jg*F  #adding the contribution from the $J_g \delta(t)$ term
    return conv

