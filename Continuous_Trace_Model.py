# -*- coding: utf-8 -*-
"""
This code implements equations from Perry's Chemical Engineers' Handbook (8th ed.) section 6-51:6-52 for calculating the drag on a particle 
Then calculates charge based upon an entered fit and calculates the resulting charge trace a Faraday cup would expect to see

Created: 11/12/23
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter



def linear_interp(x, x1, x2, y1, y2):
    
    """ Performas a linear interpolation """

    y =  y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    
    return y



def trap_int(x_array, y_array, x1, x2):
    
    """ Performs a numerical integration using the trapezium rule """
    
    total_integral = 0
    
    for i, x in enumerate(x_array):
        if x > x1 and x < x2:
            total_integral += (x - x_array[i - 1]) * (y_array[i] + y_array[i - 1]) / 2
            if x_array[i - 1] > x: # correction at beginning of trapezium
                y_x1 = linear_interp(x_array[i - 1], x1, x, y_array[i - 1], y_array[i])
                total_integral -= (x1 - x_array[i - 1]) * (y_x1 + y_array[i - 1]) / 2
    
    return total_integral



def integration_check(dp, charge_freq_den):
    
    """ Checks how close to zero the total integral of the charge frquency density is """
    
    # Find the range where charge_freq_den is minimized to maximized
    min_index = np.argmin(charge_freq_den)
    max_index = np.argmax(charge_freq_den)
    
    # Find the index of the element closest to 0 in the selected range
    closest_to_zero_index = min_index + np.argmin(np.abs(charge_freq_den[min_index:max_index]))
    midpoint = np.log10(dp[closest_to_zero_index])
    
    neg_integral = trap_int(np.log10(dp), charge_freq_den, np.min(np.log10(dp)), midpoint)
    pos_integral = trap_int(np.log10(dp), charge_freq_den, midpoint, np.max(np.log10(dp)))
    
    print(f'negative part: {round(neg_integral, 3)}, positive part: {round(pos_integral, 3)}, total integration: {round(pos_integral + neg_integral, 3)}')
    
    return



def check_modes(mode_sizes, mode_means, mode_stds):
    
    """ Checks the modes are the same length and given then sums them """
    
    if len(mode_sizes) < 1:     
        raise ValueError("No mode sizes given")

    elif len(mode_sizes) != len(mode_means) or len(mode_sizes) != len(mode_stds):   
        raise ValueError("Mode parameter lengths differ")
    
    total_size = np.sum(mode_sizes)
    
    return total_size



def calc_size_freq(d, mode_sizes, mode_means, mode_stds, total_size):
    
    """ Function to calulate the frequency denisity for a given particle size"""
    
    frequency_density = 0
    
    for number, mode_size in enumerate(mode_sizes):
        frequency_density += (mode_size / total_size) * norm.pdf(np.log10(d), np.log10(mode_means[number]), mode_stds[number])
    
    return frequency_density



def plot_size(dp, param, param_name, log=True):
    
    """ Plots a parameter agianst particle size when both given as arrays """

    plt.plot(dp, param)
    
    plt.xlabel('Particle Size / um')
    plt.ylabel(str(param_name))
    
    if log == True:
        plt.xscale('log')
    
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    
    plt.show()
    
    
    
def calc_charge(d, a, b, c):
    
    """ Calculates the charge on a particle from a quadratic fit ax^b + c given a, b and c as well as the particle size (in um)"""
    
    charge = a * d ** b + c
    
    return charge
    


def calc_CD(Re):
    
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
        CD_p = calc_CD(Re_p) # Drag coefficient of the particle
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



def get_trace(charge_freq_den, time_fit, sample_rate, trace_time):
    
    """ Returns the expected electrometer trace given the charge frequency density and time fits as arrays along with the required sample rate and times """
    
    time_step = 1 / sample_rate # time step in s
    times = np.linspace(0, trace_time, num = int(trace_time / time_step))
    trace = []
    
    if len(charge_freq_den) != len(time_fit):
        raise ValueError(f"The charge_freq_den and time_fit do not match at lengths: {len(charge_freq_den)} and {len(time_fit)}, respectively.")

    for time in times:
        
        cut_size_index = len(time_fit) # The size index to use as default if no drops fast enough
        
        for index, t in enumerate(time_fit):
            if t <= time:
                cut_size_index = linear_interp(time, time_fit[index - 1], time_fit[index], index - 1, index)
                break
       
        if cut_size_index >= len(charge_freq_den):
            trace.append(0)
        else:
            trace.append(trap_int(np.arange(len(time_fit)), charge_freq_den, cut_size_index, len(time_fit)))
    
    trace = np.array(trace)
    
    return times, trace



if __name__ == "__main__":
    
    dp = np.logspace(np.log10(0.1), np.log10(100000), num=1000, base=10.0) # Evenly logspaced dp to veiw functions

    ###########################################################################
    # Size Fit
    
    dist_type="trimodal"
    mode_sizes = 0.0217775611128209, 0.778806870226694, 5.95140876936293 # relative sizes of modes
    mode_means = 0.719588856304199, 12.2012512193786, 141.272414183094 # means od modes / um
    mode_stds = 0.0756046451662065, 0.446142546327532, 0.445809826247974 # modes standard distributions (logspace)
    
    total_size = check_modes(mode_sizes, mode_means, mode_stds)
    frequency_density = []
    
    for d in dp:
        frequency_density.append(calc_size_freq(d, mode_sizes, mode_means, mode_stds, total_size))
    
    frequency_density = np.array(frequency_density)
    # plot_size(dp, frequency_density, "Frequency Density", log=True)
    
    ###########################################################################
    # Charge Fit

    a, b, c = 0.0000242150172948039, 2, -2.61379994963604
    
    charge_fit = []
    
    for d in dp:
        charge_fit.append(calc_charge(d, a, b, c))
    
    charge_fit = np.array(charge_fit)
    # plot_size(dp, charge_fit, "Charge / 4$\pi p_{H0}\lambda^2$e",  log=False)
    
    ###########################################################################
    # Charge Frequency Density Fit
    
    precharge_ratio = 1 # Ratio of total pre-charging to self-charging
    
    charge_freq_den = frequency_density * charge_fit
    pre_charge_freq_den = frequency_density * (dp / 100) ** 2 # Prechareg scaling with SA
    self_integral = trap_int(np.arange(len(dp)), charge_freq_den, 0, len(dp))
    pre_integral = trap_int(np.arange(len(dp)), pre_charge_freq_den, 0, len(dp))
    pre_charge_freq_den *= precharge_ratio * self_integral / pre_integral
    
    # integration_check(dp, charge_freq_den)
    # plot_size(dp, charge_freq_den, "Charge Frequency Density / a.u.",  log=True)
    # plot_size(dp, pre_charge_freq_den, "Charge Frequency Density / a.u.",  log=True)
    
    ###########################################################################
    # Drop Time Fit
    
    drop_height = 0.3725 # Height of particle drop / m
    g = 9.81 # Acceleration due to gravity in m s^-2
    p_p, p_f = 2000, 1.23 # Density of particle and fluid in kg m^-3 (around 2000 for ash and 8940 for Cu)
    mu = 1.79E-5 # Viscosity of the fluid (air) in Pa s
    cfl = 1.0 # Taget CFL, ajust if numerical instabilities are encountered
    trace_time = 10 # The time the trace is recorded for, allows the small particles where there is numerical instability to be cut off
    adjust_min_t = 1 # Parameter can be used to change the minimum timestep (default = 1)

    time_fit = get_time_fit(dp, drop_height, p_p, p_f, g, mu, cfl, adjust_min_t, trace_time)
    # plot_size(dp, time_fit, "Time / s",  log=True)

    ###########################################################################
    # Trace

    sample_rate = 10.0 # Sapmle rate in Hz
    charge_multiplier = 1 # Multiplies the entire fit, including both pre- and self-charging

    times, self_trace = get_trace(charge_freq_den, time_fit, sample_rate, trace_time)
    times, pre_trace = get_trace(pre_charge_freq_den, time_fit, sample_rate, trace_time)
    self_trace *= charge_multiplier
    pre_trace *= charge_multiplier
    total_trace = self_trace + pre_trace

    plt.plot(times, self_trace, label='self-charging')
    plt.plot(times, pre_trace, label='pre-charging')
    plt.plot(times, total_trace, label='Total charging')
    
    plt.legend()
    plt.xlabel('Time / s')
    plt.ylabel('Trace / a.u.')
    

