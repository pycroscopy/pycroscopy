# -*- coding: utf-8 -*-
"""
Created on Tue Jul 04 09:21:54 2017

@author: Enrique Alejandro
Description: this library contains the core algorithms for numerical simulations

You need to have installed:
    numba -- > this can be easily installed if you have the anaconda distribution via pip install method: pip install numba
"""
from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
from numba import jit
import sys
sys.path.append('d:\github\pycroscopy')
from pycroscopy.simulation.afm_calculations import amp_phase, e_diss, v_ts


def verlet(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_L1, k_L2, k_L3, time, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt,
           fo1, fo2, fo3, f1, f2, f3):
    """This function performs verlet algorithm (central difference) for numerical integration.
    
    It integrates the differential equations of three harmonic oscillator equations (each corresponding to a distinct
    cantilever eigenmode)
    This function does not assume ideal Euler-Bernoulli scaling but instead the cantilever parameters are passed to the
    function
    The dynamics of the cantilever are assumed to be contained in the first three flexural modes
    This function will be called each simulation timestep by a main wrap around function which will contain the specific
    contact-mechanics model.
        
    Parameters:
    ----------    
    zb : float
        z equilibrium position (average tip postion with respect to the sample)
    Fo1 : float
        amplitude of the sinuosidal excitation force term (driving force) for the first eigenmode
    Fo2 : float
        amplitude of the sinuosidal excitation force term (driving force) for the second eigenmode
    Fo3 : float
        amplitude of the sinuosidal excitation force term (driving force) for the third eigenmode
    Q1 : float
        first eigenmode's quality factor
    Q2 : float
        second eigenmode's quality factor
    Q3 : float
        third eigenmode's quality factor
    k_L1 : float
        1st eigenmode's stiffness
    k_L2 : float
        2nd eigenmode's stiffness
    k_L3 : float
        3rd eigenmode's stiffness
    z1 : float
        instant 1st eigenmode deflection contribution
    z2 : float
        instant 2nd eigenmode deflection contribution
    z3 : float
        instant 3rd eigenmode deflection contribution
    v1 : float
        instant 1st eigenmode velocity
    v2 : float
        instant 2nd eigenmode velocity
    v3 : float
        instant 3rd eigenmode velocity
    z1_old : float
        instant 1st eigenmode deflection contribution corresponding to previous timestep
    z2_old : float
        instant 2nd eigenmode deflection contribution corresponding to previous timestep
    z3_old : float
        instant 3rd eigenmode deflection contribution corresponding to previous timestep
    Fts : float
        tip-sample interacting force
    dt : float
        simulation timestep
    fo1 : float
        1st eigenmode resonance frequency
    fo2 : float
        2nd eigenmode resonance frequency
    fo3 : float
        3rd eigenmode resonance frequency
    f1 : float
        1st sinusoidal excitation frequency
    f2 : float
        2nd sinusoidal excitation frequency
    f3 : float
        3rd sinusoidal excitation frequency
    
    Returns:
    -------
    tip: float
        instant tip position for new simulation timestep
    z1 : float
        instant 1st eigenmode deflection contribution for new simulation timestep
    z2 : float
        instant 2nd eigenmode deflection contribution for new simulation timestep
    z3 : float
        instant 3rd eigenmode deflection contribution for new simulation timestep
    v1 : float
        instant 1st eigenmode velocity for new simulation timestep
    v2 : float
        instant 2nd eigenmode velocity for new simulation timestep
    v3 : float
        instant 3rd eigenmode velocity for new simulation timestep
    z1_old : float
        instant 1st eigenmode deflection contribution corresponding to current timestep
    z2_old : float
        instant 2nd eigenmode deflection contribution corresponding to current timestep
    z3_old : float
        instant 3rd eigenmode deflection contribution corresponding to current timestep    
    """
    # TODO: Simplify inputs and outputs for this function. Consider wrapping up parameters for each eignenmode into an object or use lists for all k, Q, fo, etc.
    
    a1 = ( -z1 - v1/(Q1*(fo1*2*np.pi)) + ( Fo1*np.cos((f1*2*np.pi)*time) + Fo2*np.cos((f2*2*np.pi)*time) + Fo3*np.cos((f3*2*np.pi)*time)  + Fts)/k_L1  )* (fo1*2.0*np.pi)**2
    a2 = ( -z2 - v2/(Q2*(fo2*2*np.pi)) + ( Fo1*np.cos((f1*2*np.pi)*time) + Fo2*np.cos((f2*2*np.pi)*time) + Fo3*np.cos((f3*2*np.pi)*time)  + Fts)/k_L2  )* (fo2*2.0*np.pi)**2
    a3 = ( -z3 - v3/(Q3*(fo3*2*np.pi)) + ( Fo1*np.cos((f1*2*np.pi)*time) + Fo2*np.cos((f2*2*np.pi)*time) + Fo3*np.cos((f3*2*np.pi)*time)  + Fts)/k_L3  )* (fo3*2.0*np.pi)**2
    
    # Verlet algorithm (central difference) to calculate position of the tip
    z1_new = 2*z1 - z1_old + a1*pow(dt, 2)
    z2_new = 2*z2 - z2_old + a2*pow(dt, 2)
    z3_new = 2*z3 - z3_old + a3*pow(dt, 2)

    # central difference to calculate velocities
    v1 = (z1_new - z1_old)/(2*dt)
    v2 = (z2_new - z2_old)/(2*dt)
    v3 = (z3_new - z3_old)/(2*dt)
    
    # Updating z1_old and z1 for the next run
    z1_old = z1
    z1 = z1_new
    
    z2_old = z2
    z2 = z2_new
    
    z3_old = z3
    z3 = z3_new
    
    tip = z1 + z2 + z3 + zb
    return tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old

numba_verlet = jit()(verlet) #it is important to keep this line out of the effectively accelerate the function when called

def gen_maxwell_lr(G, tau, R, dt, startprint, simultime, fo1, fo2, fo3, k_m1, k_m2, k_m3, A1, A2, A3, zb, printstep = 1, Ge = 0.0, Q1=100, Q2=200, Q3=300, H=2.0e-19):
    """This function is designed for multifrequency simulation performed over a Generalized Maxwell (Wiechert) viscoelastic surface.
    
    The contact mechanics are performed over the framework of Lee and Radok (Lee, E. Ho, and Jens Rainer Maria Radok. "The contact problem for viscoelastic bodies." Journal of Applied Mechanics 27.3 (1960): 438-444.) 
    The cantilever dynamics are assumed to be contained in the first three eigenmodes. 
    The numerical integration is performed with the aid of the verlet function(defined above)
    
    Parameters:
    ---------- 
    G :  numpy.ndarray
        moduli of the springs in the Maxwell arms of a generalized Maxwell model (also called Wiechert model)
    tau: numpy.ndarray
        relaxation times of the Maxwell arms
    R : float
        tip radius
    dt : float
        simulation timestep
    fo1 : float
        1st eigenmode resonance frequency
    fo2 : float
        2nd eigenmode resonance frequency
    fo3 : float
        3rd eigenmode resonance frequency
    k_m1 : float
        1st eigenmode's stiffness
    k_m2 : float
        2nd eigenmode's stiffness
    k_m3 : float
        3rd eigenmode's stiffness
    A1 : float
        target oscillating amplitude of 1st cantilever eigenmode
    A2 : float
        target oscillating amplitude of 2nd cantilever eigenmode
    A3 : float
        target oscillating amplitude of 3rd cantilever eigenmode
    zb : float
        cantilever equilibrium position (average tip-sample distance)
    printstep : float, optional
        how often the data will be stored, default is timestep
    Ge : float, optional
        rubbery modulus, the default value is zero
    Q1 : float, optional
        first eigenmode's quality factor
    Q2 : float, optional
        second eigenmode's quality factor
    Q3 : float, optional
        third eigenmode's quality factor
    H : float, optional
        Hammaker constant
    
    Returns:
    -------  
    np.array(t_a) : numpy.ndarray
        time trace
    np.array(tip_a) : numpy.ndarray
        array containing the tip trajectory
    np.array(Fts_a) : numpy.ndarray
        array containing the tip-sample interacting force
    np.array(xb_a) : numpy.ndarray
        numpy array containing the instant position of the viscoelastic surface    
    """
    # TODO: Simplify inputs for this function. Consider useing lists for all k, Q, fo, etc.

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
      
    f1 = fo1
    f2 = fo2
    f3 = fo3
    """
    Calculating the force amplitude to achieve the given free amplitude from amplitude response of tip excited 
    oscillator
    """
    # Amplitude of 1st mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator:
    Fo1 = k_m1*A1/(fo1*2*np.pi)**2*(((fo1*2*np.pi)**2 - (f1*2*np.pi)**2 )**2 + (fo1*2*np.pi*f1*2*np.pi/Q1)**2)**0.5
    # Amplitude of 2nd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator:
    Fo2 = k_m2*A2/(fo2*2*np.pi)**2*(((fo2*2*np.pi)**2 - (f2*2*np.pi)**2 )**2 + (fo2*2*np.pi*f2*2*np.pi/Q2)**2)**0.5
    # Amplitude of 3rd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo3 = k_m3*A3/(fo3*2*np.pi)**2*(((fo3*2*np.pi)**2 - (f3*2*np.pi)**2 )**2 + (fo3*2*np.pi*f3*2*np.pi/Q3)**2)**0.5
    
    a = 0.2e-9  # interatomic distance
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
    t = 0.0  # initializing time
    Fts = 0.0
    xb = 0.0
    pb = 0.0
    pc, pc_rate = np.zeros(len(tau)), np.zeros(len(tau))
    xc, xc_rate = np.zeros(len(tau)), np.zeros(len(tau))
    alfa = 16.0/3.0*np.sqrt(R)
    # Initializing Verlet variables
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
        if tip > xb: # aparent non contact
            for i in range(len(tau)):
                sum_Gxc = sum_Gxc + G[i]*xc[i]
            if sum_Gxc/Gg > tip:  # contact, the sample surface surpassed the tip in the way up
                xb = tip
                pb = (-xb)**1.5
                for i in range(len(tau)):
                    sum_G_pb_pc = sum_G_pb_pc + G[i]*(pb - pc[i]) 
                Fts = alfa*( Ge*pb + sum_G_pb_pc )
                # get postion of dashpots
                for i in range(len(tau)):
                    pc_rate[i] = G[i]/eta[i] * (pb - pc[i])
                    pc[i] = pc[i] + pc_rate[i]*dt
                    xc[i] = -(pc[i])**(2.0/3)
            
            else: # true non-contact
                xb = sum_Gxc/Gg
                Fts = 0.0
                for i in range(len(tau)):
                    xc_rate[i] = G[i]*(xb-xc[i])/eta[i]
                    xc[i] = xc[i] + xc_rate[i]*dt
                    pc[i] = (-xc[i])**(3.0/2)     #debugging
                     
        else:  # contact region
            xb = tip
            pb = (-xb)**1.5
            for i in range(len(tau)):
                sum_G_pb_pc = sum_G_pb_pc + G[i]*(pb - pc[i]) 
            Fts = alfa*( Ge*pb + sum_G_pb_pc )
            # get postion of dashpots
            for i in range(len(tau)):
                pc_rate[i] = G[i]/eta[i] * (pb - pc[i])
                pc[i] = pc[i] + pc_rate[i]*dt
                xc[i] = -(pc[i])**(2.0/3)        
        
        # MAKING CORRECTION TO INCLUDE VDW ACCORDING TO DMT THEORY
        if tip > xb:  # overall non-contact
            Fts = -H*R/( 6.0*( (tip-xb) + a )**2 )
        else:
            Fts = Fts - H*R/(6.0*a**2)
           
    return np.array(t_a), np.array(tip_a), np.array(Fts_a), np.array(xb_a)


GenMaxwell_jit = jit()(gen_maxwell_lr)  #this line should stay outside function to allow the numba compilation and simulation acceleration work properly

def dynamic_spectroscopy(G, tau, R, dt, startprint, simultime, fo1, fo2, fo3, k_m1, k_m2, k_m3, A1, A2, A3, printstep = 1, Ge = 0.0, Q1=100, Q2=200, Q3=300, H=2.0e-19, z_step = 1):
    """This function is designed for tapping mode spectroscopy to obtain amplitude and phase curves as the cantilever is approached towards the surface.
    
    The contact mechanics are performed over the framework of Lee and Radok for viscoelastic indentation (Lee, E. Ho, and Jens Rainer Maria Radok. "The contact problem for viscoelastic bodies." Journal of Applied Mechanics 27.3 (1960): 438-444.) 
    
    Parameters:
    ---------- 
    G :  numpy.ndarray
        moduli of the springs in the Maxwell arms of a generalized Maxwell model (also called Wiechert model)
    tau: numpy.ndarray
        relaxation times of the Maxwell arms
    R : float
        tip radius
    dt : float
        simulation timestep
    fo1 : float
        1st eigenmode resonance frequency
    fo2 : float
        2nd eigenmode resonance frequency
    fo3 : float
        3rd eigenmode resonance frequency
    k_m1 : float
        1st eigenmode's stiffness
    k_m2 : float
        2nd eigenmode's stiffness
    k_m3 : float
        3rd eigenmode's stiffness
    A1 : float
        target oscillating amplitude of 1st cantilever eigenmode
    A2 : float
        target oscillating amplitude of 2nd cantilever eigenmode
    A3 : float
        target oscillating amplitude of 3rd cantilever eigenmode
    printstep : float, optional
        how often the data will be stored, default is timestep
    Ge : float, optional
        rubbery modulus, the default value is zero
    Q1 : float, optional
        first eigenmode's quality factor
    Q2 : float, optional
        second eigenmode's quality factor
    Q3 : float, optional
        third eigenmode's quality factor
    H : float, optional
        Hammaker constant
    z_step : float, optional
        cantilever equilibrium spatial step between runs. The smaller this number, the more runs but slower the simulation
    
    Returns:
    -------  
    np.array(amp) : numpy.ndarray
        array containing the reduced amplitudes at different cantilever equilibrium positions
    np.array(phase) : numpy.ndarray
        array containing the phase shifts obtained at different cantilever equilibrium positions    
    np.array(zeq) : numpy.ndarray
        array containing the approaching cantilever equilibrium positions
    np.array(Ediss) : numpy.ndarray
        array containing the values of dissipated energy
    p.array(Virial) : np.ndarray
        array containing the values of the virial of the interaction
    np.array(peakF) : np.ndarray
        array containing valued of peak force
    np.array(maxdepth) : numpy.ndarray
        array containing the values of maximum indentation
    np.array(t_a) : numpy.ndarray
        time trace
    np.array(tip_a) : numpy.ndarray
        2D array containing the tip trajectory for each run
    np.array(Fts_a) : numpy.ndarray
        2D array containing the tip-sample interacting force for each run
    np.array(xb_a) : numpy.ndarray
        2D array array containing the instant position of the viscoelastic surface for each run   
    """
    if z_step == 1:
        z_step = A1*0.05 #default value is 5% of the free oscillation amplitude
    zeq = []
    peakF = []
    maxdepth = []
    amp = []
    phase = []
    Ediss = []
    Virial = []
    
    tip_a = []
    Fts_a = []
    xb_a = []
    zb = A1*1.1
    
    while zb > 0.0:
        t, tip, Fts, xb = GenMaxwell_jit(G, tau, R, dt, startprint, simultime, fo1, fo2, fo3, k_m1, k_m2,k_m3, A1, A2, A3, zb, printstep, Ge, Q1, Q2, Q3, H)
        A,phi = amp_phase(t, tip, fo1)
        Ets = e_diss(tip, Fts, dt, fo1)
        fts_peak = Fts[np.argmax(Fts)]
        tip_depth = xb[np.argmax(tip)] -xb[np.argmin(tip)]
        Vts = v_ts(tip-zb, Fts, dt)
        
        #Attaching single values to lists
        zeq.append(zb)
        peakF.append(fts_peak)
        maxdepth.append(tip_depth)
        amp.append(A)
        phase.append(phi)
        Ediss.append(Ets)
        Virial.append(Vts)
        
        #attaching 1D arrays to lists
        tip_a.append(tip)
        Fts_a.append(Fts)
        xb_a.append(xb)
        
        zb -= z_step
    return np.array(amp), np.array(phase), np.array(zeq), np.array(Ediss), np.array(Virial), np.array(peakF), np.array(maxdepth), t, np.array(tip_a), np.array(Fts_a), np.array(xb_a)


def verlet_FS(y_t, Q1, Q2, Q3, k1, k2, k3, time, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3, Fb1=0.0, Fb2=0.0, Fb3=0.0):
    """This function performs verlet algorithm (central difference) for numerical integration of the AFM cantilever dynamics.
    
    The equations of motion are for a based excited cantilever, to be used for a static force spectroscopy simulation
    It integrates the differential equations of three harmonic oscillators (each corresponding to a distinct
    cantilever eigenmode)
    The dynamics of the cantilever are assumed to be contained in the first three flexural modes
    This function will be called each simulation timestep by a main wrap around function which will contain the specific
    contact-mechanics model.
        
    Parameters:
    ----------    
    y_t : float
        z equilibrium position (average tip postion with respect to the sample)
    Q1 : float
        first eigenmode's quality factor
    Q2 : float
        second eigenmode's quality factor
    Q3 : float
        third eigenmode's quality factor
    k1 : float
        1st eigenmode's stiffness
    k2 : float
        2nd eigenmode's stiffness
    k3 : float
        3rd eigenmode's stiffness
    time : float
        instant time of the simulation
    z1 : float
        instant 1st eigenmode deflection contribution
    z2 : float
        instant 2nd eigenmode deflection contribution
    z3 : float
        instant 3rd eigenmode deflection contribution
    v1 : float
        instant 1st eigenmode velocity
    v2 : float
        instant 2nd eigenmode velocity
    v3 : float
        instant 3rd eigenmode velocity
    z1_old : float
        instant 1st eigenmode deflection contribution corresponding to previous timestep
    z2_old : float
        instant 2nd eigenmode deflection contribution corresponding to previous timestep
    z3_old : float
        instant 3rd eigenmode deflection contribution corresponding to previous timestep
    Fts : float
        tip-sample interacting force
    dt : float
        simulation timestep
    fo1 : float
        1st eigenmode resonance frequency
    fo2 : float
        2nd eigenmode resonance frequency
    fo3 : float
        3rd eigenmode resonance frequency
    Fb1 : float, optional
        amplitude of the 1st eigenmode Brownian force (associated to thermal noise)
    Fb2 : float, optional
        amplitude of the 2nd eigenmode Brownian force (associated to thermal noise)
    Fb3 : float, optional
        amplitude of the 3rd eigenmode Brownian force (associated to thermal noise)
    
    Returns:
    -------
    tip: float
        instant tip position for new simulation timestep
    z1 : float
        instant 1st eigenmode deflection contribution for new simulation timestep
    z2 : float
        instant 2nd eigenmode deflection contribution for new simulation timestep
    z3 : float
        instant 3rd eigenmode deflection contribution for new simulation timestep
    v1 : float
        instant 1st eigenmode velocity for new simulation timestep
    v2 : float
        instant 2nd eigenmode velocity for new simulation timestep
    v3 : float
        instant 3rd eigenmode velocity for new simulation timestep
    z1_old : float
        instant 1st eigenmode deflection contribution corresponding to current timestep
    z2_old : float
        instant 2nd eigenmode deflection contribution corresponding to current timestep
    z3_old : float
        instant 3rd eigenmode deflection contribution corresponding to current timestep           
    """
    a1 = ( - z1 - v1*1.0/(fo1*2*np.pi*Q1) + y_t + Fts/k1 + Fb1/k1) *(2.0*np.pi*fo1)**2
    a2 = ( - z2 - v2*1.0/(fo2*2*np.pi*Q2) + Fts/k2 + Fb2/k2) *(2.0*np.pi*fo2)**2
    a3 = ( - z3 - v3*1.0/(fo3*2*np.pi*Q3) + Fts/k3 + Fb3/k3) *(2.0*np.pi*fo3)**2    
    
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
    
    tip = z1 + z2 + z3
    return tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old

numba_verlet_FS = jit()(verlet_FS)    

def sfs_genmaxwell_lr(G, tau, R, dt, simultime, y_dot, y_t_initial, k_m1, fo1, Ge = 0.0, Q1=100, printstep = 1, H = 2.0e-19, Q2=200, Q3=300, startprint = 1, vdw = 1):
    """This function is designed for force spectroscopy over a Generalized Maxwel surface
    
    The contact mechanics are performed over the framework of Lee and Radok, thus strictly only applies for approach portion
    
    Parameters:
    ---------- 
    G :  numpy.ndarray
        moduli of the springs in the Maxwell arms of a generalized Maxwell model (also called Wiechert model)
    tau: numpy.ndarray
        relaxation times of the Maxwell arms
    R : float
        tip radius
    dt : float
        simulation timestep
    simultime : float
        total simulation time
    y_dot: float
        approach velocity of the cantilever's base towards the sample
    y_t_initial: float
        initial position of the cantilever base with respect to the sample    
    k_m1 : float
        1st eigenmode's stiffness
    fo1 : float
        1st eigenmode resonance frequency
    Ge : float, optional
        equilibrium modulus of the material, default value is zero 
    Q1 : float, optional
        1st eigenmode quality factor
    printstep : int, optional
        if value is 1 the data will be printed with step equal to dt
    H : float, optional
        Hamaker constant
    Q2 : float, optional
        2nd eigenmode quality factor
    Q3 : float, optional
        3rd eigenmode quality factor
    startprint : float, optional
        when the simulation starts getting printed
    vdw : int, optional
        if value is 1 van der Waals forces are neglected 
    
    Returns:
    -------
    np.array(t_a) : numpy.ndarray
        time trace
    np.array(tip_a) : numpy.ndarray
        tip position in force spectroscopy simulation
    np.array(Fts_a) : numpy.ndarray
        tip-sample force interaction if force spectroscopy simulation
    np.array(xb_a) : numpy.ndarray
        viscoelastic sample position in the simulation
    np.array(defl_a) : numpy.ndarray
        cantilever deflection duing the force spectroscopy simulation
    np.array(zs_a) : numpy.ndarray
        z-sensor position (cantilever base position)    
    """
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
    
    a = 0.2e-9  #intermolecular distancePopo
    fo2 = 6.27*fo1            # resonance frequency of the second eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    fo3 = 17.6*fo1           # resonance frequency of the third eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    k_m2 = k_m1*(fo2/fo1)**2
    k_m3 = k_m1*(fo3/fo1)**2    
    if startprint == 1:  #this is the default value when the function will start saving results
        startprint = y_t_initial/y_dot    
    eta = tau*G
    Gg = Ge
    for i in range(len(tau)): #this loop looks silly but if you replace it with Gg = Ge + sum(G[:]) it will conflict with numba making, simulation very slow
        Gg = Gg + G[i]
    t_a = []
    Fts_a = []
    xb_a = []
    tip_a = []
    defl_a = []
    zs_a = []
    printcounter = 1
    if printstep == 1: #default value of tinestep
        printstep = dt
    t = 0.0  #initializing time
    Fts = 0.0
    xb = 0.0
    pb = 0.0
    pc, pc_rate = np.zeros(len(tau)), np.zeros(len(tau))
    alfa = 16.0/3.0*np.sqrt(R) #cell constant, related to tip geometry
    #Initializing Verlet variables
    z2, z3, v2, v3, z2_old, z3_old = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    v1 = y_dot
    z1 = y_t_initial
    z1_old = y_t_initial
       
    while t < simultime:
        t = t + dt
        y_t = - y_dot*t + y_t_initial   #Displacement of the base  (z_sensor position)
        tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_verlet_FS(y_t, Q1, Q2, Q3, k_m1, k_m2, k_m3, t, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3)
        defl = tip - y_t 
        if t > ( startprint + printstep*printcounter):
            defl_a.append(defl)
            zs_a.append(y_t)
            t_a.append(t)
            Fts_a.append(Fts)
            xb_a.append(xb)
            tip_a.append(tip)
            printcounter += 1       
        sum_G_pc = 0.0
        sum_G_pb_pc = 0.0
        if tip > xb: #aparent non contact
            for i in range(len(tau)):
                sum_G_pc = sum_G_pc + G[i]*pc[i]
            if sum_G_pc/Gg > tip:  #contact, the sample surface surpassed the tip
                xb = tip
                pb = (-xb)**1.5
                for i in range(len(tau)):
                    sum_G_pb_pc = sum_G_pb_pc + G[i]*(pb - pc[i]) 
                Fts = alfa*( Ge*pb + sum_G_pb_pc )
            else:  #true non contact
                pb = sum_G_pc/Gg
                xb = pb**(2.0/3)
                Fts = 0.0         
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
        
        if vdw != 1:
            #MAKING CORRECTION TO INCLUDE VDW ACCORDING TO DMT THEORY
            if tip > xb:  #overall non-contact
                Fts = -H*R/( 6.0*( (tip-xb) + a )**2 )
            else:
                Fts = Fts - H*R/(6.0*a**2)                          
    return np.array(t_a), np.array(tip_a), np.array(Fts_a), np.array(xb_a), np.array(defl_a), np.array(zs_a)    

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






