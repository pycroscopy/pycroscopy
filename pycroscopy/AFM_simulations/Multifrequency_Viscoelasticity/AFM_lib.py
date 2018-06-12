# -*- coding: utf-8 -*-
"""
Created on Tue Jul 04 09:21:54 2017

@author: Enrique Alejandro

Description: this library contains the core algortihms for tapping mode AFM simulations.

Updated February 13th 2017
"""

import numpy as np
from numba import jit


def verlet_noIdeal_EB(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_L1, k_L2, k_L3, time, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3, f1,f2,f3):
    """This function performs verlet algorithm for integration of differential equation of harmonic oscillator"""
    """for the case of sinc excitation at the tip"""
    """This function does not assume ideal Euler-Bernoulli scaling but instead the cantilever parameters are passed to the function"""    
    
    a1 = ( -z1 - v1/(Q1*(fo1*2*np.pi)) + ( Fo1*np.cos((f1*2*np.pi)*time) + Fo2*np.cos((f2*2*np.pi)*time) + Fo3*np.cos((f3*2*np.pi)*time)  + Fts)/k_L1  )* (fo1*2.0*np.pi)**2
    a2 = ( -z2 - v2/(Q2*(fo2*2*np.pi)) + ( Fo1*np.cos((f1*2*np.pi)*time) + Fo2*np.cos((f2*2*np.pi)*time) + Fo3*np.cos((f3*2*np.pi)*time)  + Fts)/k_L2  )* (fo2*2.0*np.pi)**2
    a3 = ( -z3 - v3/(Q3*(fo3*2*np.pi)) + ( Fo1*np.cos((f1*2*np.pi)*time) + Fo2*np.cos((f2*2*np.pi)*time) + Fo3*np.cos((f3*2*np.pi)*time)  + Fts)/k_L3  )* (fo3*2.0*np.pi)**2
    
    #Verlet algorithm (central difference) to calculate position of the tip
    z1_new = 2*z1 - z1_old + a1*pow(dt, 2)
    z2_new = 2*z2 - z2_old + a2*pow(dt, 2)
    z3_new = 2*z3 - z3_old + a3*pow(dt, 2)

    #central difference to calculate velocities
    v1 = (z1_new - z1_old)/(2*dt)
    v2 = (z2_new - z2_old)/(2*dt)
    v3 = (z3_new - z3_old)/(2*dt)
    
    #Updating z1_old and z1 for the next run
    z1_old = z1
    z1 = z1_new
    
    z2_old = z2
    z2 = z2_new
    
    z3_old = z3
    z3 = z3_new
    
    tip = z1 + z2 + z3 + zb
    #tip_v = v1 + v2 + v3
    return tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old
numba_noIdeal_EB = jit()(verlet_noIdeal_EB)


def verlet(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, time, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3,f1,f2,f3):
    a1 = ( -k_m1*z1 - (mass*(fo1*2*np.pi)*v1/Q1) + Fo1*np.cos((f1*2*np.pi)*time) + Fo2*np.cos((f2*2*np.pi)*time) + Fo3*np.cos((f3*2*np.pi)*time) + Fts ) /mass
    a2 = ( -k_m2*z2 - (mass*(fo2*2*np.pi)*v2/Q2) + Fo1*np.cos((f1*2*np.pi)*time) + Fo2*np.cos((f2*2*np.pi)*time) + Fo3*np.cos((f3*2*np.pi)*time) + Fts ) /mass
    a3 = ( -k_m3*z3 - (mass*(fo3*2*np.pi)*v3/Q3) + Fo1*np.cos((f1*2*np.pi)*time) + Fo2*np.cos((f2*2*np.pi)*time) + Fo3*np.cos((f3*2*np.pi)*time) + Fts ) /mass
    
    #Verlet algorithm (central difference) to calculate position of the tip
    z1_new = 2*z1 - z1_old + a1*pow(dt, 2)
    z2_new = 2*z2 - z2_old + a2*pow(dt, 2)
    z3_new = 2*z3 - z3_old + a3*pow(dt, 2)

    #central difference to calculate velocities
    v1 = (z1_new - z1_old)/(2*dt)
    v2 = (z2_new - z2_old)/(2*dt)
    v3 = (z3_new - z3_old)/(2*dt)
    
    #Updating z1_old and z1 for the next run
    z1_old = z1
    z1 = z1_new
    
    z2_old = z2
    z2 = z2_new
    
    z3_old = z3
    z3 = z3_new
    
    tip = zb + z1 + z2 + z3
    #tip_v = v1 + v2 + v3
    return tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old

numba_verlet = jit()(verlet)




def MDR_GenMaxwell_tapping(G, tau, R, dt, simultime, zb, A1, k_m1, fo1, printstep='default', Ndy = 1000, Ge = 0.0, dmax = 10.0e-9, startprint ='default', Q1=100, Q2=200, Q3=300, H=2.0e-19, A2 = 0.0, A3 = 0.0):
    """This function runs a simulation for a parabolic probe in force spectroscopy"""
    """over a generalized Maxwell surface"""
    """Output: time, tip position, tip-sample force, contact radius, and sample position"""
    G_a = []
    tau_a = []
    for i in range(len(G)): #this for loop is to make sure tau passed does not contain values lower than timestep which would make numerical integration unstable
        if tau[i] > dt*10.0:
            G_a.append(G[i])
            tau_a.append(tau[i])
    G = np.array(G_a)
    tau = np.array(tau_a)
    
    fo2 = 6.27*fo1            # resonance frequency of the second eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    fo3 = 17.6*fo1           # resonance frequency of the third eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    k_m2 = k_m1*(fo2/fo1)**2
    k_m3 = k_m1*(fo3/fo1)**2
    mass = k_m1/(2.0*np.pi*fo1)**2  
    f1 = fo1  #excited at resonance
    f2 = fo2  #excited at resonance
    f3 = fo3  #excited at resonance
    #Calculating the force amplitude to achieve the given free amplitude from amplitude response of tip excited oscillator
    Fo1 = k_m1*A1/(fo1*2*np.pi)**2*(  (  (fo1*2*np.pi)**2 - (f1*2*np.pi)**2 )**2 + (fo1*2*np.pi*f1*2*np.pi/Q1)**2  )**0.5 #Amplitude of 1st mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo2 = k_m2*A2/(fo2*2*np.pi)**2*(  (  (fo2*2*np.pi)**2 - (f2*2*np.pi)**2 )**2 + (fo2*2*np.pi*f2*2*np.pi/Q2)**2  )**0.5  #Amplitude of 2nd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo3 = k_m3*A3/(fo3*2*np.pi)**2*(  (  (fo3*2*np.pi)**2 - (f3*2*np.pi)**2 )**2 + (fo3*2*np.pi*f3*2*np.pi/Q3)**2  )**0.5    #Amplitude of 3rd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
   
    if printstep == 'default':
        printstep = 10.0*dt
    if startprint == 'default':
        startprint = 5.0*Q1*1.0/fo1
        
    eta = G*tau #bulk viscosity of the dashpot in the SLS
    #dmax = d[len(d)-1]
    amax = (R*dmax)**0.5
    y_n = np.linspace(0.0, amax, Ndy)   #1D viscoelastic foundation with specified number of elements
    dy = y_n[1] #space step
    g_y = y_n**2/R   #1D function according to Wrinkler foundation  (Popov's rule)
    #differential viscoelastic properties of each individual element
    ke = 8*Ge*dy
    k = 8*G*dy
    c = 8*eta*dy
    kg = sum(k[:]) + ke
    #end of inserting viescoelastic properties of individual elements
    
    tip_n, x_n = np.zeros(len(y_n)), np.zeros(len(y_n))    #position of the base of each SLS dependent on position
    xc_dot_n = np.zeros(( len(y_n), len(tau)))    #velocity of each dashpot
    xc_n = np.zeros(( len(y_n), len(tau)))    #position of the dashpot of each SLS dependent on time as function of position and tau
    F_n =  np.zeros(len(y_n))  #force on each SLS element
    F_a = []   #initializing total force    
    t_a = []  #initializing time
    d_a = []  #sample position, penetration
    probe_a = []   #array that will contain tip's position
    t = 0.0
    F = 0.0
    sum_kxc = 0.0
    sum_k_xb_xc = 0.0
    printcounter = 1
    if printstep == 1:
        printstep = dt
    ar_a = [] #array with contact radius
    ar = 0.0  #contact radius
    #Initializing Verlet variables
    z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    a = 0.2e-9 #interatomic distance
    while t < simultime:
        t = t + dt
        #probe = 5.0e-9-10.0e-9*np.sin(2.0*np.pi*fo1*t)  #
        probe, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_verlet(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, t, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, F, dt, fo1,fo2,fo3,f1,f2,f3)
        if probe < 0.0:  #indentation is only meaningful if probe is lower than zero position (original position of viscoelastic foundation)
            d = -1.0*probe  #forcing indentation to be positive  (this is the indentation)
        else:
            d = 0.0
        if t > (startprint + printstep*printcounter):
            F_a.append(F)
            t_a.append(t)   
            d_a.append(x_n[0])
            ar_a.append(ar)        
            probe_a.append(probe)
        
        F = 0.0  #initializing to zero before adding up the differential forces in each element
        ar = 0.0  #initializing contact radius to zero
        for n in range(len(y_n)):  #advancing in space
            tip_n[n] =  g_y[n] - d
            if tip_n[n] > 0.0: #assuring there is no stretch of elements out of the contact area
                tip_n[n] = 0.0
    
            if tip_n[n] > x_n[n]: #aparent non contact
                for i in range(len(tau)):
                    sum_kxc = sum_kxc + k[i]*xc_n[n,i]
                if sum_kxc/kg > tip_n[n]:  #contact, the sample surface surpassed the tip in the way up
                    x_n[n] = tip_n[n]
                    for i in range(len(tau)):
                        sum_k_xb_xc = sum_k_xb_xc + k[i]*(x_n[n]-xc_n[n,i])
                    F_n[n] =  - ke*x_n[n] - sum_k_xb_xc
                else:  #true non-contact
                    x_n[n] = sum_kxc/kg
                    F_n[n] = 0.0
                sum_kxc = 0.0
                sum_k_xb_xc = 0.0
            else: #contact region, tip is lower than the sample's surface
                x_n[n] = tip_n[n]
                for i in range(len(tau)):
                    sum_k_xb_xc = sum_k_xb_xc + k[i]*(x_n[n]-xc_n[n,i])
                F_n[n] = - ke*x_n[n] - sum_k_xb_xc
                sum_k_xb_xc = 0.0
            #getting position of dashpots
            for i in range(len(tau)):
                xc_dot_n[n,i] = k[i]*(x_n[n]-xc_n[n,i])/c[i]
                xc_n[n,i] = xc_n[n,i] + xc_dot_n[n,i]*dt
    
            if F_n[n] > 0.0:
                F = F + F_n[n] #getting total tip-sample force
                ar = y_n[n]   #getting actual contact radius  
        
        #MAKING CORRECTION TO INCLUDE VDW ACCORDING TO DMT THEORY
        if probe > x_n[0]:  #overall non-contact
            F = -H*R/( 6.0*( (probe-x_n[0]) + a )**2 )
        else: #overall contact
            F = F - H*R/(6.0*a**2)
        
    
    return np.array(t_a), np.array(probe_a), np.array(F_a), np.array(ar_a), np.array(d_a)



def GenMaxwell_parabolic_LR_niEB(G, tau, R, dt, startprint, simultime, fo1, fo2, fo3, k_m1, k_m2, k_m3, A1, A2, A3, zb, printstep = 1, Ge = 0.0, Q1=100, Q2=200, Q3=300, H=2.0e-19):
    """This function is for the non ideal Euler-Bernoulli case"""
    """This function is designed for tapping over a Generalized Maxwel surface"""
    """The contact mechanics are performed over the framework of Lee and Radok, thus strictly only applies for approach portion"""
    """Modified Nov 2nd 2017"""
    """Modified Dec 2nd 2017, making sure tau passed does not contain values lower than timestep which would make numerical integration unstable"""
    G_a = []
    tau_a = []
    for i in range(len(G)): #this for loop is to make sure tau passed does not contain values lower than timestep which would make numerical integration unstable
        if tau[i] > dt*10.0:
            G_a.append(G[i])
            tau_a.append(tau[i])
    G = np.array(G_a)
    tau = np.array(tau_a)
        
      
    f1 = fo1
    f2 = fo2
    f3 = fo3
    #Calculating the force amplitude to achieve the given free amplitude from amplitude response of tip excited oscillator
    Fo1 = k_m1*A1/(fo1*2*np.pi)**2*(  (  (fo1*2*np.pi)**2 - (f1*2*np.pi)**2 )**2 + (fo1*2*np.pi*f1*2*np.pi/Q1)**2  )**0.5 #Amplitude of 1st mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo2 = k_m2*A2/(fo2*2*np.pi)**2*(  (  (fo2*2*np.pi)**2 - (f2*2*np.pi)**2 )**2 + (fo2*2*np.pi*f2*2*np.pi/Q2)**2  )**0.5  #Amplitude of 2nd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo3 = k_m3*A3/(fo3*2*np.pi)**2*(  (  (fo3*2*np.pi)**2 - (f3*2*np.pi)**2 )**2 + (fo3*2*np.pi*f3*2*np.pi/Q3)**2  )**0.5    #Amplitude of 3rd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    
    a = 0.2e-9  #interatomic distance
    eta = tau*G
    Gg = Ge
    for i in range(len(tau)):
        Gg = Gg + G[i]
    t_a = []
    Fts_a = []
    xb_a = []
    tip_a = []
    printcounter = 1
    if printstep == 1:
        printstep = dt
    t = 0.0  #initializing time
    Fts = 0.0
    xb = 0.0
    pb = 0.0
    pc, pc_rate = np.zeros(len(tau)), np.zeros(len(tau))
    xc, xc_rate = np.zeros(len(tau)), np.zeros(len(tau))
    alfa = 16.0/3.0*np.sqrt(R)
    #Initializing Verlet variables
    z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    sum_Gxc = 0.0
    sum_G_pb_pc = 0.0
    
        
    while t < simultime:
        t = t + dt
        tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_noIdeal_EB(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_m1, k_m2, k_m3, t, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3, f1,f2,f3)
        
        if t > ( startprint + printstep*printcounter):
            t_a.append(t)
            Fts_a.append(Fts)
            xb_a.append(xb)
            tip_a.append(tip)
            printcounter = printcounter + 1
       
        sum_Gxc = 0.0
        sum_G_pb_pc = 0.0  
        if tip > xb: #aparent non contact
            for i in range(len(tau)):
                sum_Gxc = sum_Gxc + G[i]*xc[i]
            if sum_Gxc/Gg > tip:  #contact, the sample surface surpassed the tip in the way up
                xb = tip
                pb = (-xb)**1.5
                for i in range(len(tau)):
                    sum_G_pb_pc = sum_G_pb_pc + G[i]*(pb - pc[i]) 
                Fts = alfa*( Ge*pb + sum_G_pb_pc )
                #get postion of dashpots
                for i in range(len(tau)):
                    pc_rate[i] = G[i]/eta[i] * (pb - pc[i])
                    pc[i] = pc[i] + pc_rate[i]*dt
                    xc[i] = -(pc[i])**(2.0/3)
            
            else: #true non-contact
                xb = sum_Gxc/Gg
                Fts = 0.0
                for i in range(len(tau)):
                    xc_rate[i] = G[i]*(xb-xc[i])/eta[i]
                    xc[i] = xc[i] + xc_rate[i]*dt
                    pc[i] = (-xc[i])**(3.0/2)     #debugging
                     
        else:  #contact region
            xb = tip
            pb = (-xb)**1.5
            for i in range(len(tau)):
                sum_G_pb_pc = sum_G_pb_pc + G[i]*(pb - pc[i]) 
            Fts = alfa*( Ge*pb + sum_G_pb_pc )
            #get postion of dashpots
            for i in range(len(tau)):
                pc_rate[i] = G[i]/eta[i] * (pb - pc[i])
                pc[i] = pc[i] + pc_rate[i]*dt
                xc[i] = -(pc[i])**(2.0/3)
        
        
        #MAKING CORRECTION TO INCLUDE VDW ACCORDING TO DMT THEORY
        if tip > xb:  #overall non-contact
            Fts = -H*R/( 6.0*( (tip-xb) + a )**2 )
        else:
            Fts = Fts - H*R/(6.0*a**2)
        
           
    return np.array(t_a), np.array(tip_a), np.array(Fts_a), np.array(xb_a)




   
        