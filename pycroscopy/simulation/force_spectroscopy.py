# -*- coding: utf-8 -*-
"""
Created on Mon Feb 05 16:38:01 2018

@author: Enrique Alejandro

Description: this is the library for the force spectroscopy repository
"""

import numpy as np
from lmfit import minimize, Parameters


def percent_error(th, ex):
    "this functions calculates the percentual error"     
    error = abs((th-ex)/th)*100.0
    return error

def log_tw(de0, maxi, nn=10):
    "this function generates a frequency or time array in log scale"
    epsilon = []
    w = de0
    de = de0
    #epsilon[0] = 0
    prints = 1
    epsilon.append(de0)
    while w < maxi:
        w = w + de
        if w < maxi:
            epsilon.append(w)              
        prints = prints + 1 
        if prints == nn:
            de = de*10
            prints = 1    
    return np.array(epsilon)

def linear_fit(x, y):
    """This function receives a function and performs linear fit"""
    """Input: array of dependent variable, x. And array of dependent variable, y."""
    """Output: slope, intersect, and coefficient of determination (r^2 value) """
    """An alternative is to use: slope, intercept, r_value, p_value, std_err = stats.linregress(t, defl)"""
    m,b = np.polyfit(x, y, 1)
    mean = sum(y)/np.size(y)
    SS_tot = sum((y-mean)**2)
    SS_res = sum(   (y - (m*x+b))**2     )
    r_2 = 1.0 - SS_res/SS_tot
    return m, b, r_2

def linear_force(params, t, force):
    "this function returns the residual to perform linear fit"
    p = params.valuesdict()
    A = p['A']
    model = (A)*t
    return  (model - force) #calculating the residual

def linear_fit_Nob(x,y):
    """This function performs linear fit in the special case where intercept is zero"""
    m, b, _ = linear_fit(x,y)   #initial guess
    params = Parameters()
    params.add('A', value = m, min=0)
    result = minimize(linear_force, params, args=(x,y), method='leastsq')
    Fdot = result.params['A'].value  
    return Fdot

def log_scale(x, t, tr=0.1, st=1.0, nn = 10):
    "this receives an array and write it weighting it in logarithmic scale"
    prints = 1
    nt = len(x)
    i =0
    x_log = []
    t_log = []
    while i <nt:
        if t[i] >= prints*tr and t[i]<=st :
            x_log.append(x[i])
            t_log.append(t[i])
            prints = prints + 1
        i = i + 1
        if prints == nn:
            tr = tr*10
            prints = 1
    return np.array(x_log),np.array(t_log)



def av_dt(array):
    "this function returns the average of the timesteps in a time array"
    i = 0
    k = 0.0
    for i in range(np.size(array)-1):
        k = k + (array[i+1]-array[i])
        dt = k/(np.size(array)-1)
    return dt


def fd_log(x, f_x, nn=20, liminf = 1, suplim = 1):
    """this function returns time and f_time arrays equally spaced in logarithmic scale"""
    """Input: time starting with average dt(comming from repulsive_FD function), and f_time related to that time array"""
    if liminf ==1:
        lim_inf = round(np.log10(x[0]),2)
    else:
        lim_inf = round(np.log10(liminf),2)
    if suplim ==1:
        sup_lim = round(np.log10(x[np.size(x)-1]),2)
    else:
        sup_lim = round(np.log10(suplim),2)
    b = np.linspace(lim_inf, sup_lim, nn)
    x_log = 10.0**b
    fx_log = np.zeros(np.size(x_log))
    for j in range(1, np.size(x_log)-1):
        for i in range(np.size(x)-1):
            if (x_log[j] - x[i])*(x_log[j] - x[i+1]) < 0.0 :  #change of sign
                if (x_log[j] - x[i]) < (x_log[j] - x[i+1]):
                    x_log[j] = x[i]
                    fx_log[j] = f_x[i]
                else:
                    x_log[j] = x[i+1]
                    fx_log[j] = f_x[i+1]
    if suplim ==1 :    
        x_log[np.size(x_log)-1] = x[np.size(x)-1]        
        fx_log[np.size(x_log)-1] = f_x[np.size(x)-1]
    else:
        x_log[np.size(x_log)-1] = x[int(suplim/av_dt(x))]
        fx_log[np.size(x_log)-1] = f_x[int(suplim/av_dt(x))]
    if liminf == 1:
        x_log[0] = x[0]        
        fx_log[0] = f_x[0]
    else:
        x_log[0] = x[int(liminf/av_dt(x))]
        fx_log[0] = f_x[int(liminf/av_dt(x))]
    return x_log, fx_log




#NON LINEAR SQUARE FITTING
def func_chi(params, t, chi_exp, arms=3):
    p = params.valuesdict()
    Jg = p['Jg']
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
                else:
                    J5 = 0.0
                    tau5 = 0.0
            else:
                J4 = 0.0
                tau4 = 0.0
                J5 = 0.0
                tau5 = 0.0
        else:
            J3 = 0.0
            tau3 = 0.0
            J4 = 0.0
            tau4 = 0.0
            J5 = 0.0
            tau5 = 0.0
    else:
        J2 = 0.0
        tau2 = 0.0
        J3 = 0.0
        tau3 = 0.0
        J4 = 0.0
        tau4 = 0.0    
        J5 = 0.0
        tau5 = 0.0           
    model = (Jg+J1+J2+J3+J4+J5)*t + J1*tau1*(np.exp(-t/tau1)-1.0) \
        + J2*tau2*(np.exp(-t/tau2)-1.0)  + J3*tau3*(np.exp(-t/tau3)-1.0) + J4*tau4*(np.exp(-t/tau4)-1.0) \
        + J5*tau5*(np.exp(-t/tau5)-1.0) #+ J6*tau_c[5]*(exp(-t/tau_c[5])-1.0)
    return  (model - chi_exp) /chi_exp #calculating the residual        



def func_conv(params, t, F, t_res_fit, t_exp, tip_norm_log, arms, dt):
    """This function performs a non-linear fit based on Eq. 13 and 16 in: López‐Guerra, Enrique A., Babak Eslami, and Santiago D. Solares. Journal of Polymer Science Part B: Polymer Physics (2017)."""
    p = params.valuesdict()
    Jg = p['Jg']
    J1 = p['J1']
    tau1 = p['tau1']
    U1 = J1/tau1*np.exp(-t/tau1)
    if arms > 1:
        J2 = p['J2']
        tau2 = p['tau2']
        U2 = J2/tau2*np.exp(-t/tau2)
        if arms > 2:
            J3 = p['J3']
            tau3 = p['tau3']
            U3 = J3/tau3*np.exp(-t/tau3)
            if arms >3:
                J4 = p['J4']
                tau4 = p['tau4']
                U4 = J4/tau4*np.exp(-t/tau4)
                if arms >4:
                    J5 = p['J5']
                    tau5 = p['tau5']
                    U5 = J5/tau5*np.exp(-t/tau5)
                else:
                    J5 = 0.0
                    tau5 = 0.0
                    U5 = 0.0
            else:
                J4 = 0.0
                tau4 = 0.0
                U4 = 0.0
                J5 = 0.0
                tau5 = 0.0
                U5 = 0.0
        else:
            J3 = 0.0
            tau3 = 0.0
            U3 = 0.0
            J4 = 0.0
            tau4 = 0.0
            U4 = 0.0
            J5 = 0.0
            tau5 = 0.0
            U5 = 0.0
    else:
        J2 = 0.0
        tau2 = 0.0
        U2 = 0.0
        J3 = 0.0
        tau3 = 0.0
        U3 = 0.0
        J4 = 0.0
        tau4 = 0.0    
        U4 = 0.0
        J5 = 0.0
        tau5 = 0.0    
        U5 = 0.0
    U_t = U1+U2+U3+U4+U5
    model = np.convolve(U_t, F,mode='full')*dt
    model =  model[range(np.size(F))] + Jg*F 
    model_log, _ = log_scale(model, t, t_res_fit, t_exp)
    return  (model_log - tip_norm_log) /tip_norm_log #calculating the residual
       

def smear(x, t, tr=0.1, st=1.0):
    "this function smears an array providing as reference also time array, time resolution and total time"
    nt = len(t)
    prints = 0
    i =0
    x_smear = []
    t_smear = []
    while i < (nt):
        if t[i] >= prints*tr and t[i]<=(st+tr) :
            x_smear.append(x[i])
            t_smear.append(t[i])
            prints = prints + 1
        i = i + 1
    return np.array(x_smear), np.array(t_smear)



def nls_fit(t_simul, tip_simul, F_simul, R, t_res, t_exp, arms =3, technique=0, Jg_i = 2.0e-10, J1_i = 1.0e-9, tau1_i = 1.0e-4, J2_i = 1.0e-8, tau2_i = 1.0e-3, J3_i = 1.0e-7, tau3_i = 1.0e-2, J4_i = 1.0e-6, tau4_i = 1.0e-1, J5_i = 1.0e-5, tau5_i = 1.0e0):
    """This function performs a non-linear fit based on Eq. 14 and 15 in: López‐Guerra, Enrique A., Babak Eslami, and Santiago D. Solares. Journal of Polymer Science Part B: Polymer Physics (2017)."""
    # create a set of Parameters
    params = Parameters()
    params.add('Jg', value = Jg_i, min=0)
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
    if technique ==0:
        F_log, t_log = log_scale(F_simul, t_simul, t_res, t_exp)    
        Fdot = linear_fit_Nob(t_log, F_log)    
        chi_exp = 16.0/3*np.sqrt(R)*tip_simul**1.5/Fdot    
        chi_exp_log, t_log = log_scale(chi_exp, t_simul, t_res, t_exp)
        result = minimize(func_chi, params, args=(t_log, chi_exp_log, arms), method='leastsq')
    else:
        F, t = smear(F_simul, t_simul, t_res, t_exp)  #this arrays will be passed on the convolution and have to have the time resolution of the experiment
        tip, _ = smear(tip_simul, t_simul, t_res, t_exp)
        t_res_fit = t_res*10.0  #this is advised to be larger that time_res      
        tip_norm = 16.0/3*np.sqrt(R)*tip**1.5
        tip_norm_log, _ = log_scale(tip_norm, t, t_res_fit, t_exp)
        dt = av_dt(t)
        result = minimize(func_conv, params, args=(t, F, t_res_fit, t_exp, tip_norm_log, arms, dt), method='leastsq')
    Jg_c = result.params['Jg'].value
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
    return Jg_c, tau_c, J_c
