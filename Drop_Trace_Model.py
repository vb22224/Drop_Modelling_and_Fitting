# -*- coding: utf-8 -*-
"""
This code implements equations from Perry's Chemical Engineers' Handbook (8th ed.) section 6-51:6-52 for calculating the drag on a particle 
Then calculates charge based upon an entered fit and calulates the resulting charge trace a Faraday cup would expect to see

Created: 27/09/23
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np



def calc_CD(Re, d=0.001):
    
    """
    Function to calculate the drag coefficient (CD) for a given Renolds number (Re)
    If 350000 < Re < 10^6, so in drag crisis the function assumes a turbulent regime with added exponential smoothing
    Re should be a float, but the function will convert it to a float if possible
    """
    
    Re = float(Re)
    
    if Re < 0:
        raise ValueError("Reynolds number cannot be negative!")
    elif Re == 0:
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
        
    return CD

vector_calc_CD = np.vectorize(calc_CD) # Vectorizses the function 



def plot_CD(Re, CD):
    
    """Code that makes a plot of CD against Re when bath are input as arrays"""
    
    plt.figure(dpi=600)
    
    plt.plot(np.full(len(CD), 0.1), CD, '--', label="Stoke's to intermediate", color='grey') # Shows first regime transition
    plt.plot(np.full(len(CD), 1000), CD, '--', label="Intermediate to Newton's", color='grey')
    plt.plot(np.full(len(CD), 350000), CD, '--', label="Newton's to drag crisis", color='grey')
    plt.plot(np.full(len(CD), 1E6), CD, '--', label="Drag crisis to turbulent", color='grey')
    plt.plot(Re, CD, label='C$_D$', color='black')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Re')
    plt.ylabel('C$_D$')
    #plt.legend(loc='lower left')
    
    plt.show()



def plot_vs_size(size_array, y_array, y_axis="y"):
    
    """Plots the any array against the size array"""
    
    plt.plot(size_array, y_array, 'x')
    plt.xlabel('d$_p$ / $\mu$m')
    plt.ylabel(y_axis)
    
    
    
def plot_time_hist(time_array):
    
    """PLots a histogram of the time array"""
    
    log_bins = np.logspace(np.log10(time_array.min()), np.log10(time_array.max()), 20)
    plt.hist(time_array, bins=log_bins, edgecolor='black')
    plt.xlabel('Time to cup / s')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())



def plot_size_hist(size_array):
    
    """Code that makes a histogram of the particles size distribution from an array of sizes"""
    
    plt.figure(dpi=600)
    
    log_bins = np.logspace(np.log10(size_array.min()), np.log10(size_array.max()), 20)
    plt.hist(size_array, bins=log_bins, edgecolor='black')
    
    plt.xlabel('d$_p$ / $\mathrm{\mu}$m')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    
    plt.show()



def single_drop_time(dp, target_cfl=1.0, drop_height=0.3725, p_p=2000, p_f=1.23, g=9.81, mu=1.79E-5, plot=False):
    
    """Calculates and can plot the dropping of a single spherical particle of given size and density"""
    
    dp_m = dp * 1e-6 # Particle size in m
    vol = (4/3) * np.pi * (dp_m / 2) ** 3 # Volume of particle / m^3
    mass = vol * p_p # Mass of particle / kg
    time = 0 # Time starts at 0 s
    initial_timestep = 0.0001 # Initial timestep small then will be determined by CFL
    timestep = initial_timestep
    u, v = 0, 0 # Initial particle velocity and next step velocity is 0 m s^-2
    pos = drop_height # Initial particle position / m
    F_g = - mass * g # Force due to gravity (-ve as downwards) / N 
    area = np.pi * (dp_m / 2) ** 2 # Cross sectional area of the particle / m^2
    
    if plot == True:
        time_lis, pos_lis = [], []
        
    while pos > 0:
        if plot == True:
            time_lis.append(time)
            pos_lis.append(pos)
          
        Re_p = dp_m * abs(u) * p_f / mu # Reynolds number of the particle
        CD_p = calc_CD(Re_p) # Drag coefficient of the particle
        F_D = (1 / 2) * CD_p * area * p_f * u ** 2  # Calulates drag force / N
        
        if u != 0: # Makes sure you don't divide by 0
            F_D *= - u / abs(u) # Puts the drag in opposite direction to velocity
            
        F = F_D + F_g # Net force in the up direction / N
        accel = F / mass # Acceleration of particle in m s^-2
            
        time += timestep
        v += accel * timestep
        S = u * timestep + 1/2 * accel * timestep ** 2 # Diplacement (stepwise)
        pos += S
        u = v
        # timestep = initial_timestep
        
        if S == 0:
            timestep = initial_timestep
        else:
            timestep = target_cfl * dp_m / abs(v) # Calculating timestep to target CFL

    if plot == True:
        plt.figure(dpi=600)
        plt.xlabel('Time / s')
        plt.ylabel('Height above cup / m')
        plt.plot(time_lis, pos_lis)
    
    no_drag_time = np.sqrt(2 * drop_height/ g)
    
    if time < no_drag_time or pos > drop_height:
        print("ERROR!")
    
    return time

array_drop_time = np.vectorize(single_drop_time)



def charge_fit(dp, a_scatt=0, b_scatt=0):
    
    """Estimates a relative charge on a particle based on its size, using a
    quadratic fit where a and b are scattered by a normal distribution"""
    
    a, b = 0.000633983, -1.009141454
    a += np.random.normal(loc=0, scale=a_scatt, size=1)[0]
    b += np.random.normal(loc=0, scale=b_scatt, size=1)[0]
    charge = a * dp ** 2 + b
    
    return charge

array_charge_fit = np.vectorize(charge_fit)



def normalise_charge(charge_array, size_array):
    
    """ Function used to remove any net charge from a charge array """
    
    Total_charge = np.sum(charge_array)
    
    # Shifting all particles evenly
    # charge_array -= Total_charge / len(charge_array) 
    
    # Shisting particles based on surface area
    Total_dp_squared = np.sum(size_array ** 2)
    charge_array -= Total_charge * size_array ** 2 / Total_dp_squared
    
    return charge_array



def get_trace(time_array, charge_array, size_array, precharge_ratio=1, time_length=20, step = 0.1, plot=True):
    
    """Produces the expected measured charge trace from the time and charge array"""
    
    time = np.arange(0, time_length + step, step)
    selfcharge_trace, precharge_trace, total_trace = [], [], []
    
    abs_charge = np.sum(abs(charge_array)) # Total charge
    precharge_array = size_array ** 2
    abs_precharge = np.sum(abs(precharge_array))
    precharge_array *= abs_charge * precharge_ratio / abs_precharge
    
    for t1 in time:
        cumulative_charge_self, cumulative_charge_pre = 0, 0
        for count, t2 in enumerate(time_array):
            if t2 < t1:
                cumulative_charge_self += charge_array[count]
                cumulative_charge_pre += precharge_array[count]
        selfcharge_trace.append(cumulative_charge_self)
        precharge_trace.append(cumulative_charge_pre)
        cumulative_total = cumulative_charge_pre + cumulative_charge_self
        total_trace.append(cumulative_total)
        
    if plot == True:
        plt.figure(dpi=600)
        plt.plot(time, selfcharge_trace, label='self-charging') 
        plt.plot(time, precharge_trace, label='pre-charging') 
        plt.plot(time, total_trace, label='total charging') 
        plt.xlabel('Time / s')
        plt.ylabel('Charge / a.u. ')
        plt.legend()
        plt.show()
        
    return total_trace, time



def plot_trace_list(precharge_ratio_list=[1], time_length=20):
    
    "Plots a figure containing traces with different precharge ratios"

    precharge_ratio_list = sorted(precharge_ratio_list, reverse=True) # Sorts the list decending so that the legend will be nicer
    plt.figure(dpi=600)
    
    for pr in precharge_ratio_list:
        trace, time = get_trace(time_array, charge_array, size_array, precharge_ratio=pr, time_length=time_length, step=0.01, plot=False)
        plt.plot(time, trace, label=str(pr))
    
    plt.xlabel('Time / s')
    plt.ylabel('Charge / a.u. ')
    plt.legend(loc='lower right')
    plt.show()
    
    return

        

if __name__ == "__main__":
    
    Np = 1000 # Number of Particles
    dist_mean, dist_sd = 30, 0.2 # Mean and standard distribution from the lognormal particle distribution
    target_cfl = 1 # Ideally 1
    drop_height = 0.3725 # Height of particle drop / m
    g = 9.81 # Acceleration due to gravity in m s^-2
    p_p, p_f = 8940, 1.23 # Density of particle and fluid in kg m^-3 (around 2000 for ash and 8940 for Cu)
    mu = 1.79E-5 # Viscosity of the fluid (air) in Pa s
    precharge_ratio = 1.0 # Ratio of total pre-charging to self-charging
    
    size_array = 10 ** np.random.normal(loc=np.log10(dist_mean), scale=dist_sd, size=Np) # Generates particle sizes / um
    # plot_size_hist(size_array) # Plots a histogram of particle sizes

    # Re = np.logspace(-4, 7, num=1000)  # Generates logarithmically spaced array
    # CD = vector_calc_CD(Re) # Calculates Re
    # plot_CD(Re, CD) # Plots CD against Re
    
    # dp = 5 # Particel size in um
    # single_drop_time(dp, target_cfl=target_cfl, drop_height=drop_height, p_p=p_p, plot=True)
    
    time_array = array_drop_time(size_array, target_cfl=target_cfl, drop_height=drop_height, p_p=p_p)
    charge_array = array_charge_fit(size_array, a_scatt=0.0002, b_scatt=0.3) # Typical vaues: a_scatt=0.0002, b_scatt=0.3
    charge_array = normalise_charge(charge_array, size_array)
    
    # plot_vs_size(size_array, time_array, "Time / s")
    # plot_time_hist(time_array)
    # plot_vs_size(size_array, charge_array, "Charge / 4$\pi p_{H0}\lambda^2$e")

    # trace = get_trace(time_array, charge_array, size_array, precharge_ratio, time_length=20, step=0.01, plot=True)
    
    precharge_ratio_list = [5, 2, 1, 0, -1, -2, -5]
    # precharge_ratio_list = [1, 0.5, 0, -0.5, -1]
    plot_trace_list(precharge_ratio_list, time_length=20)
    


    
    
    