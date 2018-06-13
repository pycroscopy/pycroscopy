# -*- coding: utf-8 -*-
"""
Created on Tue Jul 04 09:21:54 2017

@author: Enrique Alejandro
Description: this library contains the core algortihms for tapping mode AFM simulations.

You need to have installed:
    numba -- > this can be easily installed if you have the anaconda distribution via pip install method: pip install numba
"""

import numpy as np
from numba import jit


def verlet(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_L1, k_L2, k_L3, time, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3, f1,f2,f3):
    """This function performs verlet algorithm (central difference) for numerical integration of the differential equations of three harmonic oscillator equations (each corresponding to a distinct cantilever eigenmode)
    This function does not assume ideal Euler-Bernoulli scaling but instead the cantilever parameters are passed to the function
    The dynamic of the cantilever are assumed to be contained in the first three flexural modes.
    This function will be called each simulation timestep by a main wrap around function which will contain the specific contact-mechanics model
    
    
    Args:
        zb: z equilibrium postion (average tip postion with respect to the sample)
        Fo1, Fo2, Fo3: amplitude of the sinuosidal excitation force terms (driving force) for each of 1st 3 eigenmodes
        Q1, Q2, Q3: three first eigenmodes' quality factor
        k_L1, k_L2, k_L3: three first eigenmodes' stiffnesses
        time: instant time of simulation
        z1, z2, z3: instant eigenmode trajectory
        v1,v2,v3: instant eigenmode velocity
        z1_old, z2_old, z3_old: position of the 1st three eigenmodes corresponding to previous timestep
        Fts: tip-sample interacting force
        dt: simulation timestep
        fo1,fo2,fo3: first three eigenmodes' resonance freuquencies
        f1,f2,f3: excitation frequencies (often chosen to be equal to the eigenmodes' frequencies)
    
    Returns:
        tip: instant tip position for new simulation timestep
        z1, z2, z3: eigenmodes' positions updated for new timestep
        v1, v2, v3: eigenmodes' velocities calculated for new timestep
        z1_old, z2_old, z3_old: eigenmodes' positions for previous timestep  
    
    """    
    
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
    return tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old
numba_verlet = jit()(verlet)



def GenMaxwell_LR(G, tau, R, dt, startprint, simultime, fo1, fo2, fo3, k_m1, k_m2, k_m3, A1, A2, A3, zb, printstep = 1, Ge = 0.0, Q1=100, Q2=200, Q3=300, H=2.0e-19):
    """This function is designed for multifrequency simulations performed over a Generalized Maxwell (Wiechert) viscoelastic surface
    The contact mechanics are performed over the framework of Lee and Radok (Lee, E. Ho, and Jens Rainer Maria Radok. "The contact problem for viscoelastic bodies." Journal of Applied Mechanics 27.3 (1960): 438-444.) 
    The cantilever dynamics are assumed to be contained in the first three eigenmodes. 
    The numerical integration is performed with the aid of the verlet function(defined above)
    
    Args:
        G: moduli of the springs in the Maxwell arms of a generalized Maxwell model (also called Wiechert model)
        teu: relaxation times of the Maxwell arms
        R: tip radius
        dt: simulation timestep
        fo1,fo2,fo3: first three eigenmodes' resonance freuquencies
        k_m1, k_m2, k_m3: three first eigenmodes' stiffnesses
        A1, A2, A3: target oscillating amplitude of each cantilever eigenmode
    
    Returns:
        np.array(t_a) : numpy array containing the time trace
        np.array(tip_a): numpy array containing the tip trajectory
        np.array(Fts_a): numpy array containing the tip-sample interacting force
        np.array(xb_a): numpy array containing the instant position of the viscoelastic surface    
    """
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
        tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_verlet(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_m1, k_m2, k_m3, t, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3, f1,f2,f3)
        
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




   
        