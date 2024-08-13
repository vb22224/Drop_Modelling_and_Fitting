# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:04:45 2024
Comparison of the different methods for calculating particle fall times
@author: vb22224
"""

import numpy as np
import matplotlib.pyplot as plt



def calc_CD(Re, time_op='Chen'):
    
    """
    Function to calculate the drag coefficient (CD) for a given Renolds number (Re)
    Will follow the Perry or Chen models
    In the Paery model: 350000 < Re < 10^6, so in drag crisis the function assumes a turbulent regime with added exponential smoothing
    Re should be a float, but the function will convert it to a float if possible
    """
    
    Re = float(Re)
    
    if time_op == 'Perry':
        if Re < 0:
            raise ValueError("Reynolds number cannot be negative!")
        elif Re == 0: # No drag at 0 speed
            CD = 0
        elif Re <= 0.1: # Stokes regime
            CD = 24 / Re
        elif Re <= 1000: # Intermediate regime
            CD = (24 / Re) * (1 + 0.14 * (Re ** 0.7))
        elif Re <= 350000: # Newton regime
            CD = 0.445
        elif Re <= 1E6: # Drag crisis assumed Newton but with a smoothing term # Shouldnt be too relevant as our particles shouldn't enter this regime
            CD = 0.19 - (80000 / Re) + 0.445 / ((Re - 350000) ** (1 / 8)) - 0.084 # Keeps all CD positive
        else: # Turbulent regime
            CD = 0.19 - (80000 / Re)
            
    elif time_op == 'Chen': # The slightly different Chen model
        if Re == 0:
            CD = 0
        elif Re < 1:
            CD = 24 / Re
        else:
            CD = 24 / Re + 3 / np.sqrt(Re) + 0.34
        
    else:
        raise ValueError(f"time_op should be 'Perry' or 'Chen', but insead: {time_op}")
    
    return CD



def fall_time(d, drop_height=0.3725, p_p=2000, p_f=1.23, g=9.81, mu=1.79E-5, cfl=1.0, min_timestep=0.0001):
    
    """ Calculates the time taken for a particle to fall a certain height, may be unstable at low d """
    
    d_m = d * 1e-6 # Particle size in m
    area = np.pi * (d_m / 2) ** 2 # Cross sectional area of the particle / m^2
    vol = (4/3) * np.pi * (d_m / 2) ** 3 # Volume of particle / m^3
    mass = vol * p_p # Mass of particle / kg
    F_g = - mass * g # Force due to gravity (-ve as downwards) / N 
    
    time = 0 # Time starts at 0 s
    
    timestep = min_timestep
    u, v = 0, 0 # Initial particle velocity and next step velocity is 0 m s^-1
    pos = drop_height # Initial particle position / m
    
    while pos > 0:
        
        Re_p = d_m * abs(u) * p_f / mu # Reynolds number of the particle
        CD_p = calc_CD(Re_p, 'Perry') # Drag coefficient of the particle
        F_D = (1 / 2) * CD_p * area * p_f * u ** 2  # Calulates drag force / N
        
        if u != 0: # Makes sure you don't divide by 0
            F_D *= - u / abs(u) # Puts the drag in opposite direction to velocity
            
        F = F_D + F_g # Net force in the up direction / N
        accel = F / mass # Acceleration of particle in m s^-2
       
        # Update time, velocity, and position
        time += timestep
        v += accel * timestep
        S = u * timestep + 1/2 * accel * timestep ** 2 # Diplacement (stepwise)
        pos += S
        u = v
        
        if S == 0:
            timestep = min_timestep
        else:
            timestep = cfl * d_m / abs(v) # Calculating timestep to target CFL
            if timestep < min_timestep:
                timestep = min_timestep

    # print(f"For a particle of {d} um the time to fall {drop_height} m is {time} s.")

    return time



def get_time_fit(dp, drop_height=0.3725, p_p=2000, p_f=1.23, g=9.81, mu=1.79E-5, cfl=1.0, adjust_min_tp=1, trace_time=10):
    
    """
    With help of the fall_time function calculates the numpy time array for particles that fall within the trace time
    There are a few arbritary ajustments to try to keep numerical stability for physically revelvant particle sizes
    """
    
    time_fit = []
    
    for d in np.flip(dp):
        
        if d < 20: #arbitary but decreasing cfl at low d helps with numerical stability
            cfl /= 10
            
        min_timestep = adjust_min_tp * 0.0008 * (d / 100) # This is arbritraty but found to work at p_p = 2000 kg m^-3
        time_fit.append(fall_time(d, drop_height, p_p, p_f, g, mu, cfl, min_timestep))
        
        if time_fit[-1] > trace_time:
            break
    
    while len(time_fit) < len(dp):
        time_fit = np.append(time_fit, time_fit[-1])
        
    time_fit = np.flip(np.array(time_fit))
    
    # dragless_time = np.sqrt(2 * drop_height / g)
    # print(f"Drop height expected for no drag: {dragless_time}")
    
    return time_fit



def chen_time_fit(dp, drop_height=0.3725, p_p=2000, p_f=1.23, g=9.81, mu=1.79E-5, cfl=1.0, adjust_min_tp=1, trace_time=10):
    
    """
    Utilising Chen and Fryrear's approach to caclulate fall time encorporating aerodynamic diameter
    """
    
    time_fit = []
    
    for d in np.flip(dp):
        
        r = d / 2000000 # Radius in m
        dt = 0.0001 # Timestep
        Vs, Z, t = 0, 0, 0    
        
        while Z < drop_height:
            
            Re_p = p_f * Vs * r * 2 / mu # Reynolds number of the particle
            Cd = calc_CD(Re_p, 'Chen') # Drag coefficient of the particle
            dVs = (g - (3 * p_f * Cd * Vs ** 2) / (4 * p_p * r * 2))
            Z += Vs * dt + 0.5 * dVs * dt ** 2
            Vs += dVs * dt
            
            if Z >= drop_height:
                time_fit.append(t)
            else:
                t += dt
        
        if time_fit[-1] > trace_time:
            break
        
    while len(time_fit) < len(dp):
        time_fit = np.append(time_fit, time_fit[-1])
        
    time_fit = np.flip(np.array(time_fit))
    
    return time_fit



def dp_to_da(dp, p_p=2000, p_f=1.23, g=9.81, mu=1.79E-5):
    
    """ Converts geometric (seiving) diamter to aerodynamic diamter empiracally using work by Chen et al. """
    
    da, Re_array = [], []
    
    for d in dp:
        r = d / 2000000 # Radius in m
        Vs = (2 * p_p * g * r ** 2) / (9 * mu) # Terminal velocity
        Re = p_f * Vs * r * 2 / mu # Reynolds number
        # da.append((1.32 - 0.21 * np.log(Re)) * d) # Adding the converted diamter
        da.append(6.704 * d ** 0.586)
        Re_array.append(Re)
        
    return np.array(da), np.array(Re_array)



if __name__ == "__main__":
    
    dp = np.linspace(0.1, 600, 1000)
    
    drop_height = 0.3725 # Height of particle drop / m
    g = 9.81 # Acceleration due to gravity in m s^-2
    p_p, p_f = 2670, 1.225 # Density of particle and fluid in kg m^-3 (around 1500 for ash, 2670 for Laradorite, 8940 for Cu, and 1290 for MGS-1)
    mu = 1.79E-5 # Viscosity of the fluid (air) in Pa s
    cfl = 1.0 # Taget CFL, ajust if numerical instabilities are encountered
    trace_time = 60 # The time the trace is recorded for, allows the small particles where there is numerical instability to be cut off
    adjust_min_t = 0.1 # Parameter can be used to change the minimum timestep (default = 1)
    
    # Aerodynamic diamter conversion
    da, Re_array = dp_to_da(dp, p_p, p_f, g, mu)
    
    # # Old is the time caculated using Perry's model
    # old_time_fit = get_time_fit(dp, drop_height, p_p, p_f, g, mu, cfl, adjust_min_t, trace_time)
    # old_time_da_fit = get_time_fit(da, drop_height, p_p, p_f, g, mu, cfl, adjust_min_t, trace_time)
    
    # # New is the time calculated using Chen and Fryrear's approach
    # new_time_fit = chen_time_fit(dp, drop_height, p_p, p_f, g, mu, cfl, adjust_min_t, trace_time)
    # new_time_da_fit = chen_time_fit(da, drop_height, p_p, p_f, g, mu, cfl, adjust_min_t, trace_time)
    
    # # Time with no drag
    # flat = np.array([0, np.max(dp)])
    # dragless_time = np.full((2, 1), np.sqrt(2 * drop_height / g))

    
    # Ploting diamter conversion against Reynolds number
    # plt.figure(dpi=200)
    # plt.plot(Re_array, da / dp)
    # plt.xlabel('Re')
    # plt.ylabel('d$_a$ / d$_g$')
    # plt.xlim([0, 70])
    # plt.ylim([0, 3])
    
    # Plotting diamter conversion
    plt.figure(dpi=200, figsize=(3, 3))
    plt.plot(dp, da)
    plt.plot([0, max(dp)], [0, max(dp)], '--', color='black')
    plt.xlabel('d$_g$ / $\mu$m')
    plt.ylabel('d$_a$ / $\mu$m')
    plt.ylim([0, 400])
    plt.xlim([0, 600])
    
    # # Plotting time fits
    # plt.figure(dpi=200)
    # plt.plot(dp, old_time_fit, '-', label='Perry model')
    # plt.plot(dp, old_time_da_fit, '-', label='Perry model - aerodynamic ajustment')
    # plt.plot(dp, new_time_fit, '-', label='Chen model')
    # plt.plot(dp, new_time_da_fit, '-', label='Chen model')
    # plt.plot(flat, dragless_time, '--', color='black', label='Dragless')
    # plt.xlabel('d$_p$ / $\mu$m')
    # plt.ylabel('Time / s')
    # plt.xlim([0, np.max(dp)])
    # plt.xlim([0, 100])
    # plt.ylim([0, 30])
    
    # # Plotting all 
    
    # plt.legend()
    # plt.show()
    
    
    
    