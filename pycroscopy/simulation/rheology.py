# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:20:30 2018

@author: Enrique Alejandro

Description:  This library contains useful rheology based functions, for example for the interconversion
between generalized Maxwell and Voigt models, calculator of operator coefficients from Maxwell or Voigt 
parameters.
"""

import numpy as np
from lmfit import minimize, Parameters
import sys
sys.path.append('d:\github\pycroscopy')
from pycroscopy.simulation.afm_calculations import av_dt


def compliance_maxwell(G, tau , Ge = 0.0, dt = 1, simul_t = 1, lw=0):
    """This function returns the numerical compliance of a Generalized Maxwell model.
    
    This numerical compliance is useful for interconversion from Gen Maxwell model to generalized Voigt model
    
    Parameters:
    ---------- 
    G :  numpy.ndarray
        moduli of the springs in the Maxwell arms of a generalized Maxwell model (also called Wiechert model)
    tau: numpy.ndarray
        relaxation times of the Maxwell arms
    Ge : float, optional
        equilibrium modulus of the material, default value is zero 
    dt : float, optional
        simulation timestep
    simul_t : float, optional
        total simulation time
    lw : int, optional
        flag to return calculated compliance with logarithmic weight
    
    Returns:
    ---------- 
    np.array(t_r) : numpy.ndarray
        array containing the time trace
    np.array(J_r) : numpy.ndarray
        array containing the calculated creep compliance   
    """
    if dt == 1:  #if timestep is not user defined it will given as a fracion of the lowest characteristic time
        dt = tau[0]/100.0
    if simul_t ==1: #if simulation time is not defined it will be calculated with respect to largest retardation time
        simul_t = tau[len(tau)-1]*10.0e3
            
    G_a = []
    tau_a = []
    """
    this for loop is to make sure tau passed does not contain values lower than time step which would make numerical 
    integration unstable
    """
    for i in range(len(G)):
        if tau[i] > dt*10.0:
            G_a.append(G[i])
            tau_a.append(tau[i])
    G = np.array(G_a)
    tau = np.array(tau_a)
    Gg = Ge
    for i in range(len(tau)): #this loop looks silly but if you replace it with Gg = Ge + sum(G[:]) it will conflict with numba making, simulation very slow
        Gg = Gg + G[i]
    eta = tau*G
    Jg =1.0/Gg  #glassy compliance
    N = len(tau)
    
    Epsilon_visco = np.zeros(N) #initial strain
    Epsilon_visco_dot = np.zeros(N) #initial strain velocity
        
    t_r = []  #creating list with unknown number of elements
    J_r = []  #creating list with unknown number of elements
    time = 0.0
    J_t = Jg #initial compliance
    print_counter = 1
    tr = dt  #printstep
    
    while time < simul_t: #CREEP COMPLIANCE SIMULATION, ADVANCING IN TIME
        time = time + dt
        sum_Gn_EpsVisco_n = 0.0   #this sum has to be resetted to zero every timestep
        for n in range(0,N):
            Epsilon_visco_dot[n] = G[n]*(J_t - Epsilon_visco[n])/eta[n]
            Epsilon_visco[n] = Epsilon_visco[n] + Epsilon_visco_dot[n]*dt
            sum_Gn_EpsVisco_n = sum_Gn_EpsVisco_n + G[n]*Epsilon_visco[n]
        J_t = (1 + sum_Gn_EpsVisco_n)/Gg 
        if time >= print_counter*tr and time < simul_t:
            t_r.append(time)
            J_r.append(J_t)
            print_counter += 1
        if lw != 0:  #if logarithmic weight is activated, the data will be appended weighted logarithmically
            if print_counter == 10:
                tr = tr*10
                print_counter = 1
          
    return np.array(t_r), np.array(J_r)         

def relaxation_voigt(J, tau, Jg, phi_f = 0.0, dt = 1, simul_t = 1, lw = 0):
    """This function returns the numerical relaxation modulus of a Generalized Voigt model
        
    This numerical relaxation modulus is useful for interconversion from Gen Maxwell model to generalized Voigt model
    
    Parameters:
    ---------- 
    J :  numpy.ndarray
        compliances of the springs in the Voigt units of a generalized Voigt model
    tau: numpy.ndarray
        relaxation times of the Maxwell arms
    Jg : float
        glassy compliance of the material 
    dt : float, optional
        simulation timestep
    simul_t : float, optional
        total simulation time
    lw : int, optional
        flag to return calculated compliance with logarithmic weight
    
    Returns:
    ---------- 
    np.array(t_r) : numpy.ndarray
        array containing the time trace
    np.array(G_r) : numpy.ndarray
        array containing the calculated relaxation modulus   
    """    
    if dt == 1:  #if timestep is not user defined it will given as a fracion of the lowest characteristic time
        dt = tau[0]/100.0
    if simul_t ==1: #if simulation time is not defined it will be calculated with respect to largest retardation time
        simul_t = tau[len(tau)-1]*10.0e3
    
    J_a = []
    tau_a = []
    """
    this for loop is to make sure tau passed does not contain values lower than time step which would make numerical 
    integration unstable
    """
    for i in range(len(J)):
        if tau[i] > dt*10.0:
            J_a.append(J[i])
            tau_a.append(tau[i])
    J = np.array(J_a)
    tau = np.array(tau_a)
    
    Gg = 1.0/Jg
    N = len(tau)
    phi = J/tau
    #Defining initial conditions
    x = np.zeros(N)
    x_dot = np.zeros(N)
    t_r = []  #creating list with unknown number of elements
    G_r = []  #creating list with unknown number of elements
    time = 0.0
    G_t = Gg #initial relaxation modulus
    print_counter = 1
    tr = dt #printstep
    
    while time < simul_t: #RELAXATION MODULUS SIMULATION, ADVANCING IN TIME
        time = time + dt
        k = len(tau) - 1
        while k > -1:
            if k == len(tau) - 1:
                x_dot[k] = G_t*phi[k]
            else:
                x_dot[k] = G_t*phi[k] + x_dot[k+1]
            k -=1
        for i in range(len(tau)):
            x[i] = x[i] + x_dot[i]*dt
        G_t = Gg*(1.0-x[0])
        if time >= print_counter*tr and time <simul_t:
            t_r.append(time)
            G_r.append(G_t)
            print_counter += 1
        if lw != 0: #if logarithmic weight is activated, the data will be appended weighted logarithmically
            if print_counter == 10:
                tr = tr*10
                print_counter = 1
    
    return np.array(t_r), np.array(G_r) 
    
    
def comp_fit(params, t, compliance, Jg, arms=3):
    """This function contains the fitting model that will be passed to the non linear least square process
    
    This function will be called by the main wrap around funtion: comp_fit_main. This model will be compared with simulation or experimental data to retrieve generalized Voigt parameters from creep data
    
    Parameters:
    ---------- 
    params :  numpy.ndarray
        parameters defined by the non-linear optimization wrap around function: comp_fit_main
    t: numpy.ndarray
        time trace
    compliance: numpy.ndarray
        array contaning the creep compliance data
    Jg : float
        glassy compliance of the material
    arms : int, optional
        number of Voigt units used in the fitting model
        
    Returns:
    ---------- 
    (model - compliance)/compliance : float
        residual to be minimized by lmfit.minimize in comp_fit_main function    
    """
    p = params.valuesdict()
    J1 = p['J1']
    tau1 = p['tau1']
    if arms > 1:
        J2 = p['J2']
        tau2 = p['tau2']
        if arms > 2:
            J3 = p['J3']
            tau3 = p['tau3']
            if arms >3:
                J4 = p['J4']
                tau4 = p['tau4']
                if arms >4:
                    J5 = p['J5']
                    tau5 = p['tau5']
                    if arms >5:
                        J6 = p['J6']
                        tau6 = p['tau6']
                        if arms >6:
                            J7 = p['J7']
                            tau7 = p['tau7']
                            if arms >7:
                                J8 = p['J8']
                                tau8 = p['tau8']
                            else:
                                J8 = 0.0
                                tau8=0.0
                        else:
                            J7=0.0
                            tau7=0.0
                            J8 = 0.0
                            tau8=0.0
                    else:
                        J6 = 0.0
                        tau6 = 0.0
                        J7=0.0
                        tau7=0.0
                        J8 = 0.0
                        tau8=0.0
                else:
                    J5 = 0.0
                    tau5 = 0.0
                    J6 = 0.0
                    tau6 = 0.0
                    J7=0.0
                    tau7=0.0
                    J8 = 0.0
                    tau8=0.0
            else:
                J4 = 0.0
                tau4 = 0.0
                J5 = 0.0
                tau5 = 0.0
                J6 = 0.0
                tau6 = 0.0
                J7=0.0
                tau7=0.0
                J8 = 0.0
                tau8=0.0
            
        else:
            J3 = 0.0
            tau3 = 0.0
            J4 = 0.0
            tau4 = 0.0
            J5 = 0.0
            tau5 = 0.0
            J6 = 0.0
            tau6 = 0.0
            J7=0.0
            tau7=0.0
            J8 = 0.0
            tau8=0.0
    else:
        J2 = 0.0
        tau2 = 0.0
        J3 = 0.0
        tau3 = 0.0
        J4 = 0.0
        tau4 = 0.0    
        J5 = 0.0
        tau5 = 0.0  
        J6 = 0.0
        tau6 = 0.0
        J7=0.0
        tau7=0.0
        J8 = 0.0
        tau8=0.0         
    model =  Jg + ( J1*( 1.0 - np.exp(-t/tau1) ) + J2*(1.0 - np.exp(-t/tau2) )  + \
    J3*( 1.0 - np.exp(-t/tau3) ) + J4*( 1.0 - np.exp(-t/tau4)) + J5*(1.0- np.exp(-t/tau5) )  + \
    J6*( 1.0 - np.exp(-t/tau6) ) + J7*( 1.0 - np.exp(-t/tau7)) + J8*(1.0- np.exp(-t/tau8) )  )  # according to eq. 3.5-16 in Tschoegl book. #+J6*exp(-t/tau6)+J7*exp(-t/tau7) )
    return  (model - compliance)/compliance #calculating the residual     


def comp_fit_main(t, compliance, Jg, arms =3, J1_i = 1.0e-9, tau1_i = 1.0e-4, J2_i = 1.0e-8, tau2_i = 1.0e-3, J3_i = 1.0e-7, tau3_i = 1.0e-2, J4_i = 1.0e-6, tau4_i = 1.0e-1, J5_i = 1.0e-5, tau5_i = 1.0e0, J6_i = 1.0e-5, tau6_i = 1.0e0, J7_i = 1.0e-5, tau7_i = 1.0e0, J8_i = 1.0e-5, tau8_i = 1.0e0):
    """Wrap around function for the non-linear fitting to get generalized voigt parameters from generalized maxwell parameters
    
    This function uses the auxiliary comp_fit function defined above for the minimization of error
    
    Parameters:
    ---------- 
    t: numpy.ndarray
        time trace
    compliance: numpy.ndarray
        array contaning the creep compliance data
    Jg : float
        glassy compliance of the material
    arms : int, optional
        number of Voigt units used in the fitting model
    J1_i : float, optional
        compliance of spring in 1st voigt unit
    tau1_i : float, optional
        retardation time of 1st voigt unit
    J2_i : float, optional
        compliance of spring in 2nd voigt unit
    tau2_i : float, optional
        retardation time of 2nd voigt unit
    J3_i : float, optional
        compliance of spring in 3rd voigt unit
    tau3_i : float, optional
        retardation time of 3rd voigt unit
    J4_i : float, optional
        compliance of spring in 4th voigt unit
    tau4_i : float, optional
        retardation time of 4th voigt unit
    J5_i : float, optional
        compliance of spring in 5th voigt unit
    tau5_i : float, optional
        retardation time of 5th voigt unit
    J6_i : float, optional
        compliance of spring in 6th voigt unit
    tau6_i : float, optional
        retardation time of 6th voigt unit
    J7_i : float, optional
        compliance of spring in 7th voigt unit
    tau7_i : float, optional
        retardation time of 7th voigt unit
    J8_i : float, optional
        compliance of spring in 8th voigt unit
    tau8_i : float, optional
        retardation time of 8th voigt unit        
        
    Returns:
    ---------- 
    tau_c : float
        retrieved retardation times corresponding to generalied Voigt model
    J_c : float
        retrieved values of compliances of springs in the Voigt units of the generalized Voigt model        
    """    
    params = Parameters() # creating a set of Parameters for the fitting model
    params.add('J1', value = J1_i, min=0)
    params.add('tau1', value = tau1_i, min=tau1_i/10.0, max=tau1_i*10.0)
    if arms > 1:
        params.add('J2', value = J2_i, min=0)
        params.add('tau2', value = tau2_i, min=tau2_i/10.0, max=tau2_i*10.0)
        if arms >2:
            params.add('J3', value = J3_i, min=0)
            params.add('tau3', value = tau3_i, min=tau3_i/10.0, max=tau3_i*10.0)
            if arms>3:
                params.add('J4', value = J4_i, min=0)
                params.add('tau4', value = tau4_i, min=tau4_i/10.0, max=tau4_i*10.0)
                if arms>4:
                    params.add('J5', value = J5_i, min=0)
                    params.add('tau5', value = tau5_i, min=tau5_i/10.0, max=tau5_i*10.0)
                    if arms>5:
                        params.add('J6', value = J5_i, min=0)
                        params.add('tau6', value = tau5_i, min=tau5_i/10.0, max=tau5_i*10.0)
                        if arms>6:
                            params.add('J7', value = J5_i, min=0)
                            params.add('tau7', value = tau5_i, min=tau5_i/10.0, max=tau5_i*10.0)
                            if arms>7:
                                params.add('J8', value = J5_i, min=0)
                                params.add('tau8', value = tau5_i, min=tau5_i/10.0, max=tau5_i*10.0)
                
    result = minimize(comp_fit, params, args=(t, compliance, Jg, arms), method='leastsq')
    J_c= np.zeros(arms)  #N is the number of voigt units retrieved
    tau_c = np.zeros(arms)
    J_c[0] = result.params['J1'].value
    tau_c[0]= result.params['tau1'].value
    if arms > 1:
        J_c[1] =result.params['J2'].value
        tau_c[1]= result.params['tau2'].value
        if arms >2:
            J_c[2] =result.params['J3'].value
            tau_c[2]= result.params['tau3'].value
            if arms>3:
                J_c[3] =result.params['J4'].value
                tau_c[3]= result.params['tau4'].value
                if arms>4:
                    J_c[4] =result.params['J5'].value
                    tau_c[4]= result.params['tau5'].value
                    if arms>5:
                        J_c[5] =result.params['J6'].value
                        tau_c[5]= result.params['tau6'].value
                        if arms>6:
                            J_c[6] =result.params['J7'].value
                            tau_c[6]= result.params['tau7'].value
                            if arms>7:
                                J_c[7] =result.params['J8'].value
                                tau_c[7]= result.params['tau8'].value
    return tau_c, J_c


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

def theta_vG(omega, Jg, J, tau, phi = 0.0):    
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

