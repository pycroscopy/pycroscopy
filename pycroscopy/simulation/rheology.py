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


def compliance_maxwell(G, tau , Ge = 0.0, dt = 1, simul_t = 1):
    """This function returns the numerical compliance of a Generalized Maxwell model"""
    """This numerical compliance may be useful for interconversion from Gen Maxwell model"""
    """to generalized Voigt model"""
    if dt == 1:
        dt = tau[0]/100.0
    if simul_t ==1:
        simul_t = tau[len(tau)-1]*10.0e3
    if dt > tau[0]:   #MAKING SURE TIMESTEP IS NOT SMALLER THAN SMALLEST RELAXATION TIME TO AVOID THE SIMULATION TO BLOW UP
        tau_i = tau
        G_i = G
        tau = []
        G = []
        for i in range(len(tau_i)):
            if tau_i[i] > dt*10.0:
                tau.append(tau_i[i])
                G.append(G_i[i])
        tau = np.array(tau)
        G = np.array(G)
    Gg = sum(G[:]) + Ge
    eta = tau*G
    #DEFINING GLASSY COMPLIANCE AND RUBBERY COMPLIANCE
    Jg =1.0/Gg
    N = len(tau)
    #DEFINING INITIAL CONDITIONS
    Epsilon_visco = np.zeros(N) #initial strain
    Epsilon_visco_dot = np.zeros(N) #initial strain velocity
        
    t_r = []  #creating list with unknown number of elements
    J_r = []  #creating list with unknown number of elements
    time = 0.0
    J_t = Jg #initial compliance
    print_counter = 1
    tr = dt
    #CREEP SIMULATION, ADVANCING IN TIME
    while time < simul_t:
        time = time + dt
        sum_Gn_EpsVisco_n = 0.0   #this sum has to be resetted to zero every timestep
        for n in range(0,N):
            Epsilon_visco_dot[n] = G[n]*(J_t - Epsilon_visco[n])/eta[n]
            Epsilon_visco[n] = Epsilon_visco[n] + Epsilon_visco_dot[n]*dt
            sum_Gn_EpsVisco_n = sum_Gn_EpsVisco_n + G[n]*Epsilon_visco[n]
        J_t = (1 + sum_Gn_EpsVisco_n)/Gg 
        if time >= print_counter*tr and time <simul_t:
            t_r.append(time)
            J_r.append(J_t)
            print_counter += 1
        #if print_counter == 10:
        #    tr = tr*10
        #    print_counter = 1
          
    return np.array(t_r), np.array(J_r)         

def relaxation_voigt(J, tau, Jg, phi_f = 0.0, dt = 1, simul_t = 1):
    """This function returns the numerical relaxation modulus of a Generalized Voigt model"""
    if dt == 1:
        dt = tau[0]/100.0
    if simul_t ==1:
        simul_t = tau[len(tau)-1]*10.0e3
    #Je = sum(J[:]) + Jg
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
    #epsilon_g = 0.0
    print_counter = 1
    tr = dt
    #RELAXATION MODULUS SIMULATION, ADVANCING IN TIME
    while time < simul_t:
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
        if print_counter == 10:
            tr = tr*10
            print_counter = 1
    
    return np.array(t_r), np.array(G_r) 
    
    
def comp_fit(params, t, compliance, Jg, arms=3):
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
    """This function performs a non-linear fit to get generalized voigt parameters from generalized maxwell parameters"""
    # create a set of Parameters
    params = Parameters()
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
    "this function gives an array of storage compliance on omega"
    J_prime = np.zeros(len(omega))
    for i in range(len(omega)):
        if len(J) > 1:
            J_prime[i] = Jg + sum( J[:]/(1.0 + (  pow(omega[i],2)*pow(tau[:],2) ) ) )
        else:
            J_prime[i] = Jg + ( J/(1.0 + (  pow(omega[i],2)*pow(tau,2) ) ) )
            
    return J_prime

def j_loss(omega, Jg, J, tau, phi = 0):
    "this function gives an array of J_loss on omega"
    J_biprime = np.zeros(len(omega))
    for i in range(len(omega)):
        if np.size(J)>1:
            J_biprime[i] = sum( J[:]*omega[i]*tau[:]/(1.0 + (pow(omega[i],2)*pow(tau[:],2)) ) ) + phi/omega[i]
        else:
            J_biprime[i] = ( J*omega[i]*tau/(1.0 + (pow(omega[i],2)*pow(tau,2)) ) ) + phi/omega[i]
    return J_biprime

def theta_vG(omega, Jg, J, tau, phi =0):    
    "this function returns the loss angle from Generalized Kelvin Voigt Prony Coefficients"
    Jloss = j_loss(omega, Jg, J, tau, phi =0)
    Jstorage =  j_storage(omega, Jg, J, tau)
    theta = np.arctan(Jloss/Jstorage)*180/np.pi
    return theta

def j_abs(G_storage, G_loss):
    "returns the absolute compliance"
    J_absolute = np.sqrt(j_loss**2+j_storage**2)
    return J_absolute

def g_loss(omega, G, tau, Ge = 0.0):
    "this function returns the value of G_loss for either a point value or an array of omega"
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
    "this function returns the value of G_store for either a point value or an array of omega"
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
    "this function returns the loss angle from Generalized Maxwell Prony Coefficients"
    Gloss = g_loss(omega, G, tau, Ge)
    Gstorage = g_storage(omega, G, tau, Ge)
    theta = np.arctan(Gloss/Gstorage)*180.0/np.pi
    return theta       

def g_abs(G_storage, G_loss):
    "returns the absolute modulus"
    G_absolute = np.sqrt(G_loss**2+G_storage**2)
    return G_absolute


def chi_th(t, Jg, J, tau, phi = 0):
    "this function gives the strain response to a unit slope stress (the time varying fluidity)"
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
    "this function returns the compliance in time t, for a model with given set of parameters"
    if (np.size(J)) > 1:
        Je = sum(J[:])+Jg
    else:
        Je = J+Jg
    comp = np.zeros(len(t))
    for i in range (len(t)):
        if len(J)>1:
            comp[i] = Je - sum(J[:]*np.exp(-t[i]/tau[:])) + phi*t[i]
        else:
            comp[i] = Je - (J*np.exp(-t[i]/tau)) + phi*t[i]
    return comp

def g_t(t, G, tau, Ge = 0.0):
    """this function returns the relaxation modulus in time"""
    G_rel = np.zeros(np.size(t))
    if np.size(G) == 1:  #the model is the SLS
        for i in range(np.size(t)):
            G_rel[i] = Ge + G*np.exp(-t[i]/tau)
    else: #the model has more than one arm
        for i in range(np.size(t)):
            G_rel[i] = Ge + sum(G[:]*np.exp(-t[i]/tau[:]))
    return G_rel

def u_t(time, J, tau, phi=0):
    "this function gives the response of a unit strain impulse"
    "It does not contain the term with the delta function"
    U = np.zeros(len(time))
    for i in range(len(time)):
        U[i] = sum(J[:]/tau[:]*np.exp(-time[i]/tau[:])) + phi
    return U

def conv_uf(time, F, Jg, J, tau, phi=0):
    "this function convolves force and the retardance U(t)"
    dt = av_dt(time)
    U = np.zeros(len(time))
    for i in range(len(time)):
        U[i] = sum(J[:]/tau[:]*np.exp(-time[i]/tau[:])) +phi
    conv = np.convolve(U, F, mode='full')*dt
    conv = conv[range(len(F))] + Jg*F
    return conv

