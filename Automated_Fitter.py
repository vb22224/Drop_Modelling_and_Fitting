# -*- coding: utf-8 -*-
"""
Code to fit a modelled charge trace to a measured distribution

Created: 23/11/23
"""

import numpy as np
import matplotlib.pyplot as plt
import Drop_Plotter as dp
import Drop_Trace_Model as dtm
import Continuous_Trace_Model as ctm
from scipy.optimize import minimize



def trace_plotter(time, total_trace, av_times, averaged_data, selfcharge_trace=[], precharge_trace=[]):
    
    """ Function to plot various modelled and measured traces """
    
    plt.figure(dpi=600)

    if selfcharge_trace is not None and len(selfcharge_trace) > 0:
        plt.plot(time, selfcharge_trace, '-', label='self-charging', linewidth=1.0) 

    if precharge_trace is not None and len(precharge_trace) > 0:
        plt.plot(time, precharge_trace, '-', label='pre-charging', linewidth=1.0) 
        
    plt.plot(time, total_trace, '-', label='predicted trace', linewidth=1.0) 
    plt.plot(av_times, averaged_data, '-', label='measured trace', linewidth=1.0) 
    plt.xlabel('Time / s')
    plt.ylabel('Charge / a.u. ')
    plt.legend()
    
    plt.show()
    
    
    
def shift_prediction(time, total_trace, av_times, averaged_data):
    
    """ Function to shift the precdicted charge trace for the induction ring probes so that it can be overlaid with the measured trace """
    
    exp_min_time = av_times[averaged_data.index(min(averaged_data))]
    mod_min_time = time[total_trace.index(min(total_trace))]
    exp_max, mod_max = max(averaged_data, key=abs), max(total_trace, key=abs)
    print(f"q off by a factor of: {round(mod_max / exp_max, 3)}")

    for count, t in enumerate(time):
        time[count] -= (mod_min_time - exp_min_time)
        total_trace[count] /= (mod_max / exp_max)

    return time, total_trace
    
    
    
def get_total_residual(time, total_trace, av_times, averaged_data, selfcharge_trace, precharge_trace, crop=True, sync_trace=True):
    
    """ Function to compute the total residual between the experimental and residual trace """
    
    total_residual = 0
    
    if sync_trace == True:
        for count, t in enumerate(total_trace):
            if t != 0:
                av_times += time[int(count/2)] # tries to account for the "dead-time" in the predicted signal (may be better without this)
                break
    
    if crop == True:
        
        if max(time) <= max(av_times):
            raise ValueError("The predicted timelength is less than the experimental time")
        
        for number, t in enumerate(time): # Cropping the predicted trace
            if t >  max(av_times):
                time = time[:number + 1]
                total_trace = total_trace[:number + 1]
                selfcharge_trace = selfcharge_trace[:number + 1]
                precharge_trace = precharge_trace[:number + 1]
                break
    
    for number, d in enumerate(averaged_data):
        
        index = np.abs(time - av_times[number]).argmin()
        residual = np.abs(d - total_trace[index])
        total_residual += residual
    
    return total_residual, av_times, time, selfcharge_trace, precharge_trace, total_trace
    
    
    
def predict_trace(charge_multiplier, precharge_ratio, charge_freq_den, pre_charge_freq_den, self_integral, pre_integral, time_fit, sample_rate, trace_time):
    
    """ Predicts the expected traces at specified precharge ratio and charge multiplier """
    
    pre_charge_freq_den *= precharge_ratio * self_integral / pre_integral
    time, selfcharge_trace = ctm.get_trace(charge_freq_den, time_fit, sample_rate, trace_time)
    time, precharge_trace = ctm.get_trace(pre_charge_freq_den, time_fit, sample_rate, trace_time)
    selfcharge_trace *= charge_multiplier
    precharge_trace *= charge_multiplier
    total_trace = selfcharge_trace + precharge_trace
    
    return time, selfcharge_trace, precharge_trace, total_trace
    
    
    
def objective_function(variables, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    
    """ Used for the function minimisation with scipy.minimize """
    
    x1, x2 = variables[0], variables[1] # charge_multiplier, precharge_ratio
    
    result = function_to_minimize(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11)
      
    return result

        
    
def function_to_minimize(charge_multiplier, precharge_ratio, charge_freq_den, pre_charge_freq_den, self_integral, pre_integral, time_fit, sample_rate, trace_time, av_times, averaged_data):
    
    """ 
    Function that calculates the trace and residuals from the predict_trace function
    Then gets the total residual using get_total_residual
    """
    
    time, selfcharge_trace, precharge_trace, total_trace = predict_trace(charge_multiplier, precharge_ratio, charge_freq_den, pre_charge_freq_den, self_integral, pre_integral, time_fit, sample_rate, trace_time)
    total_residual, av_times, time, selfcharge_trace, precharge_trace, total_trace = get_total_residual(time, total_trace, av_times, averaged_data, selfcharge_trace, precharge_trace, crop=True)
    print(f"The total residual for this fit is: {round(total_residual, 1)} pC")
    
    return total_residual


    
if __name__ == "__main__":
    
    ###########################################################################
    # Parameters required for the Drop_Plotter
    
    path = "..\\Drops\\Fuego_Fit"
    input_type = "dir" # Directory (dir), categorised (cat), list (lis), or list of lists (lol)
    # Note: the categroised (cat) input type requires filenames formatted such that the type is follwed by underscore then index e.g. Atitlan_1.txt
    plot_parameter = "Q" # Charge (Q) or voltage (V)
    change_type = "mm" # How the boxplot is measured: by the change from the start to finish (sf), by the diference between min and max (mm), or by mm - abs(sf) (mmsf)
    plot_option = "Trace" # Box plot (Box), The trace (Trace), or both (Both)
    ignore_len_errors = "Extend" # Should be "Crop" or "Extend" if you want to shorten data to shortest series or extend to the longest "Error" returns error. Note: if "Crop" selected this will affect teh data plotted in the boxplot too
    plot_average = True # If True plots the average trace instead of all individual traces
    file_names_lis = ['Ring_1.txt', 'Ring_2.txt', 'Ring_3.txt', 'Ring_4.txt', 'Ring_5.txt']
    file_names_lol = [["Atitlan_1.txt", "Atitlan_2.txt", "Atitlan_3.txt", "Atitlan_4.txt", "Atitlan_5.txt"],
                      ["Atitlan_6.txt", "Atitlan_7.txt", "Atitlan_8.txt", "Atitlan_9.txt", "Atitlan_10.txt"]]
    remove_files = ["Atitlan_2.txt", "Atitlan_4.txt", "Atitlan_6.txt", "Atitlan_11.txt", "Atitlan_15.txt", "StHelens_1.txt", "Fuego_3.txt"]
    lol_labels = ["Min", "Max"]
    trim = True # If True removes the begining of the trace such that all traces start at the same time (required to calulate average trace)
    manual_trim = {"Cu 1 mm_3.txt": 2.25, "Cu 1 mm_5.txt": 2.95, "Cu 1 mm_6.txt": 3, "Cu 1 mm_7.txt": 4} # Dictionary of file names and the time (in seconds) that you want to maually trim from the start
    store_dict, read_dict = False, False # Options to store or read from file the manual trim dictionary
    
    file_names, lol_structure, lol_labels = dp.get_file_names(input_type, path, file_names_lis, file_names_lol, remove_files, lol_labels)
    data, sample_rates, temperatures, humidities = dp.read_and_convert_data(file_names, path, plot_parameter, ignore_len_errors, trim, manual_trim, store_dict, read_dict)
    time_step, times, time = dp.get_times(sample_rates, data)
    av_times, averaged_data = dp.plot_figure(data, times, time, input_type, plot_parameter, file_names, path, lol_structure, lol_labels, change_type, plot_option, plot_average, show=False, get_av_data=True)
    
    ###########################################################################
    # Parameters required for the Continuous_Trace_Model

    dp = np.logspace(np.log10(0.1), np.log10(10000), num=1000, base=10.0) # Evenly logspaced dp to veiw functions
    
    drop_height = 0.3725 # Height of particle drop / m
    g = 9.81 # Acceleration due to gravity in m s^-2
    p_p, p_f = 1500, 1.23 # Density of particle and fluid in kg m^-3 (around 2000 for ash and 8940 for Cu)
    mu = 1.79E-5 # Viscosity of the fluid (air) in Pa s
    cfl = 1.0 # Taget CFL, ajust if numerical instabilities are encountered
    trace_time = 10 # The time the trace is recorded for, allows the small particles where there is numerical instability to be cut off
    adjust_min_t = 1 # Parameter can be used to change the minimum timestep (default = 1)
    sample_rate = sample_rates[0] / 100 # Can devide this by less (or 1) to increse accracy but also time to compute
    initial_guess = [-0.002, 1000] # Charge multiplier and precharge ratio initial guesses
    
    # Size Fit
    dist_type="trimodal"
    mode_sizes = 0.0217775611128209, 0.778806870226694, 5.95140876936293 # relative sizes of modes
    mode_means = 0.719588856304199, 12.2012512193786, 141.272414183094 # means od modes / um
    mode_stds = 0.0756046451662065, 0.446142546327532, 0.445809826247974 # modes standard distributions (logspace)
    
    # Charge Fit
    a, b, c = 0.0000242150172948039, 2, -2.61379994963604
    
    total_size = ctm.check_modes(mode_sizes, mode_means, mode_stds)
    frequency_density, charge_fit = [], []
    
    for d in dp:
        frequency_density.append(ctm.calc_size_freq(d, mode_sizes, mode_means, mode_stds, total_size))
        charge_fit.append(ctm.calc_charge(d, a, b, c))
    
    frequency_density = np.array(frequency_density)
    charge_fit = np.array(charge_fit)
    
    charge_freq_den = frequency_density * charge_fit
    pre_charge_freq_den = frequency_density * (dp / 100) ** 2 # Prechareg scaling with SA
    self_integral = ctm.trap_int(np.arange(len(dp)), charge_freq_den, 0, len(dp))
    pre_integral = ctm.trap_int(np.arange(len(dp)), pre_charge_freq_den, 0, len(dp))
    time_fit = ctm.get_time_fit(dp, drop_height, p_p, p_f, g, mu, cfl, adjust_min_t)
    
    # charge_multiplier, precharge_ratio = initial_guess[0], initial_guess[1]
    # time, selfcharge_trace, precharge_trace, total_trace = predict_trace(charge_multiplier, precharge_ratio, charge_freq_den, pre_charge_freq_den, self_integral, pre_integral, time_fit, sample_rate, trace_time)
    # total_residual, av_times, time, selfcharge_trace, precharge_trace, total_trace = get_total_residual(time, total_trace, av_times, averaged_data, selfcharge_trace, precharge_trace, crop=True)
    # trace_plotter(time, total_trace, av_times, averaged_data, selfcharge_trace, precharge_trace)
    ###########################################################################
    # Fit
    
    result = minimize(objective_function, initial_guess, method=None, args=(charge_freq_den, pre_charge_freq_den, self_integral, pre_integral, time_fit, sample_rate, trace_time, av_times, averaged_data))
    charge_multiplier, precharge_ratio = result.x[0], result.x[1]
    # charge_multiplier, precharge_ratio = -0.91536396526753179, 0
    
    print(f"Charge multiplier: {charge_multiplier}, precharge_ratio: {precharge_ratio}")
    
    time, selfcharge_trace, precharge_trace, total_trace = predict_trace(charge_multiplier, precharge_ratio, charge_freq_den, pre_charge_freq_den, self_integral, pre_integral, time_fit, sample_rate, trace_time)
    total_residual, av_times, time, selfcharge_trace, precharge_trace, total_trace = get_total_residual(time, total_trace, av_times, averaged_data, selfcharge_trace, precharge_trace, crop=True)
    trace_plotter(time, total_trace, av_times, averaged_data, selfcharge_trace, precharge_trace)
    
    ###########################################################################
    # Section for ring probe
    
    # target_cfl = 0.001
    # p_p, p_f = 30, 1.23 # Density of particle and fluid in kg m^-3 (around 2000 for ash and 8940 for Cu)
    # ring_height = drop_height / 3 # Height of the ring probe / m
    # dp = 4610 # Particle size in um
    # charge = -35 # Charge on particel in pC
    # time_lis, ring_trace = dtm.single_ring_trace(dp, charge, target_cfl=target_cfl, drop_height=drop_height, ring_height=ring_height, p_p=p_p, plot=False)
    # time_lis, ring_trace = shift_prediction(time_lis, ring_trace, av_times, averaged_data)
    # trace_plotter(time_lis, ring_trace, av_times, averaged_data)


    
