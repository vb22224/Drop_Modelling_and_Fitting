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
import Continuous_Fitter as cf
from scipy.optimize import minimize



def trace_plotter(time, total_trace, av_times, averaged_data, selfcharge_trace=[], precharge_trace=[], average_minus_error=[], average_plus_error=[], title=None, dpi=600, error_lines=False, adjust_by_mass=False):
    
    """ Function to plot various modelled and measured traces """
    
    plt.figure(dpi=dpi) # dpi=600

    if selfcharge_trace is not None and len(selfcharge_trace) > 0:
        plt.plot(time, selfcharge_trace, '-', label='self-charging', linewidth=1.0) 

    if precharge_trace is not None and len(precharge_trace) > 0:
        plt.plot(time, precharge_trace, '-', label='pre-charging', linewidth=1.0) 
    
    if error_lines:
        plt.plot(av_times, average_minus_error, '--', dashes=(15,15), linewidth=0.5, color='r')
        plt.plot(av_times, average_plus_error, '--', dashes=(15,15), linewidth=0.5, color='r')
    
    plt.plot(time, total_trace, '-', label='predicted trace', linewidth=1.0) 
    plt.plot(av_times, averaged_data, '-', label='measured trace', linewidth=1.0, color='r') 
    plt.xlabel('Time / s')
    if adjust_by_mass:
        plt.ylabel('Specific Charge / pC g$^{-1}$')
    else:
        plt.ylabel('Charge / a.u. ')
    plt.legend()
    
    if title != None:
        plt.title(title)
    
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
    
    
    
def predict_trace(charge_multiplier, precharge_ratio, charge_freq_den, pre_charge_freq_den, self_integral, pre_integral, time_fit, sample_rate, trace_time):
    
    """ Predicts the expected traces at specified precharge ratio and charge multiplier """
    
    pre_charge_freq_den *= precharge_ratio * self_integral / pre_integral
    time, selfcharge_trace = ctm.get_trace(charge_freq_den, time_fit, sample_rate, trace_time)
    # plt.plot(pre_charge_freq_den)
    time, precharge_trace = ctm.get_trace(pre_charge_freq_den, time_fit, sample_rate, trace_time)
    selfcharge_trace *= charge_multiplier
    precharge_trace *= charge_multiplier
    total_trace = selfcharge_trace + precharge_trace

    
    return time, selfcharge_trace, precharge_trace, total_trace
    
    
    
def objective_function(variables, *args):
    
    """ Used for the function minimisation with scipy.minimize """
    
    x3, x4, x5, x6, x7, x8, x9, x10, x11, minimise_op, adjust_by_mass = args
    x4 = np.array(x4)
    x1, x2, time_shift = variables[0], variables[1], variables[2]
    TR, R2 = function_to_minimize(adjust_by_mass, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, time_shift)
    
    if minimise_op == "TR":
        result = TR
    elif minimise_op == "R2":
        result = 1 - R2
    else:
        raise ValueError(f"minimise_op should be either total residual (TR), or R² value (R2). Not: {minimise_op}.")
    
    return result

        
    
def function_to_minimize(adjust_by_mass, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, time_shift):
    
    """ 
    Function that calculates the trace and residuals from the predict_trace function
    Then gets the total residual calling the function: get_total_residual
    """
    
    time, selfcharge_trace, precharge_trace, total_trace = predict_trace(y1, y2, y3, y4, y5, y6, y7, y8, y9)
    # plt.plot(time, precharge_trace)
    total_residual, y10, time, selfcharge_trace, precharge_trace, total_trace, R2 = cf.get_total_residual(time, total_trace, y10, y11, selfcharge_trace, precharge_trace, time_shift=time_shift, crop=True)
    
    info = f"cm: {round(y1, 4)}, pr: {round(y2, 4)}, ts: {time_shift} tr: {round(total_residual, 0)}, r2: {round(R2, 3)}"
    # print(info)
    # Comment the following plotter to spped up fit
    # trace_plotter(time, total_trace, y10, y11, selfcharge_trace, precharge_trace, title = info, dpi=100, adjust_by_mass=adjust_by_mass)
    
    print(f"[{y1}, {y2}, {time_shift}]")
    print(f"Total residual: {round(total_residual, 1)} pC, R² value: {round(R2, 3)}")
    
    return total_residual, R2


    
if __name__ == "__main__":
    
    ###########################################################################
    # Parameters required for the Drop_Plotter
    
    path = "..\\Drops\\Sand"
    input_type = "cat" # Directory (dir), categorised (cat), list (lis), or list of lists (lol)
    # Note: the categroised (cat) input type requires filenames formatted such that the type is follwed by underscore then index e.g. Atitlan_1.txt
    plot_parameter = "Q" # Charge (Q) or voltage (V)
    change_type = "mm" # How the boxplot is measured: by the change from the start to finish (sf), by the diference between min and max (mm), or by mm - abs(sf) (mmsf)
    plot_option = "Trace" # Box plot (Box), The trace (Trace), or both (Both)
    ignore_len_errors = "Extend" # Should be "Crop" or "Extend" if you want to shorten data to shortest series or extend to the longest "Error" returns error. Note: if "Crop" selected this will affect teh data plotted in the boxplot too
    plot_average = True # If True plots the average trace instead of all individual traces
    file_names_lis = ['Lab2Small_1.txt']
    file_names_lol = [["Atitlan_1.txt", "Atitlan_2.txt", "Atitlan_3.txt", "Atitlan_4.txt", "Atitlan_5.txt"],
                      ["Atitlan_6.txt", "Atitlan_7.txt", "Atitlan_8.txt", "Atitlan_9.txt", "Atitlan_10.txt"]]
    remove_files = ['63-75_1.txt', '63-75_4.txt', '63-125_5.txt', '63-125_4.txt']
    lol_labels = ["Min", "Max"]
    spec_cat = ['CAlAin'] # specify categories to include if input type is categorised (file names before the underscore), leave empty to include all
    
    trim = True # If True removes the begining of the trace such that all traces start at the same time (required to calulate average trace)
    error_lines = 'N' # Shows the error when averaged, should be area (A), line (L), or none (N)
    manual_trim = {"Cu 1 mm_3.txt": 2.25, "Cu 1 mm_5.txt": 2.95, "Cu 1 mm_6.txt": 3, "Cu 1 mm_7.txt": 4} # Dictionary of file names and the time (in seconds) that you want to maually trim from the start
    store_dict, read_dict = False, True # Options to store or read from file the manual trim dictionary
    adjust_by_mass = True
    base_time, tolerance_amp, tolerance_len = 0.5, 1.5, 300
    adjust_for_decay, time_const = True, 600 # in [s] and comes from the conductivity of air (David's = 310.5)
    data_col = 0 # Selects which column of data file to read (default is the first column = 0)

    file_names, lol_structure, lol_labels = dp.get_file_names(input_type, path, file_names_lis, file_names_lol, remove_files, lol_labels, spec_cat=spec_cat)
    data, sample_rates, temperatures, humidities, filtered_names = dp.read_and_convert_data(file_names, path, plot_parameter, ignore_len_errors, base_time, tolerance_amp, tolerance_len, trim=trim, manual_trim=manual_trim, store_dict=store_dict, read_dict=read_dict, data_col=data_col, adjust_by_mass=adjust_by_mass, adjust_for_decay=adjust_for_decay, time_const=time_const)
    time_step, times, time = dp.get_times(sample_rates, data)
    av_times, averaged_data, average_minus_error, average_plus_error = dp.plot_figure(data, times, time, input_type, plot_parameter, file_names, path, lol_structure,lol_labels, change_type, plot_option, plot_average=plot_average,
                                                                                      show=False, get_av_data=True, get_error=True, adjust_by_mass=adjust_by_mass)
    
    ###########################################################################
    # Parameters required for the Continuous_Trace_Model

    dp = np.logspace(np.log10(0.1), np.log10(10000), num=1000, base=10.0) # Evenly logspaced dp to veiw functions
    
    drop_height = 0.3725 # Height of particle drop / m
    g = 9.81 # Acceleration due to gravity in m s^-2
    p_f = 1.225 # Density of fluid in kg m^-3
    mu = 1.79E-5 # Viscosity of the fluid (air) in Pa s
    cfl = 1.0 # Taget CFL, ajust if numerical instabilities are encountered
    trace_time = 10 # The time the trace is recorded for, allows the small particles where there is numerical instability to be cut off
    adjust_min_t = 0.1 # Parameter can be used to change the minimum timestep (default = 1)
    sample_rate = sample_rates[0] / 10 # Can devide this by less (or 1) to increse accracy but also time to compute
    minimise_op = "R2" # Minimisation option, either total residual (TR), or R² value (R2)
    initial_guess = [0.02, -2, -0.1] # Charge multiplier, precharge ratio, and time shift initial guesses
    
    # Size Fit
    convert_size = True # If True will convert the distribution from a volume distribution to number
    mode_sizes = [0.12047909541723714, 0.8699151732927874]
    mode_means = [69.41390100092396, 155.46376875563044]
    mode_stds = [0.09710018329900653, 0.1262506307566037]
    truncate = float('inf')
    
    # Charge Fit
    fit_type = 'complex' # Simple or complex depending on the shape of the output. Complex only needs a and b
    a, b, c = 0.001736898030812294, -1285.9991540916203, 0
    
    # Time Fit
    p_p = 2670 # Density of particle in kg m^-3 (around 1500 for ash, 2670 for Laradorite and 8940 for Cu)
    trace_time = 10 # The time the trace is recorded for, allows the small particles where there is numerical instability to be cut off
    convert_time = ['E', 0.87, 8.797, 0.541] # [convert?, constant, alpha, beta] First option none (N), constant (C) or exponential (E) for whether to convert the distribution and the fitting paramters if so
    
    frequency_density, charge_fit = cf.get_size_charge(fit_type, dp, a, b, c, mode_sizes, mode_means, mode_stds, truncate, convert_size)
    frequency_density = np.array(frequency_density)
    charge_fit = np.array(charge_fit)
    
    charge_freq_den = frequency_density * charge_fit
    pre_charge_freq_den = frequency_density * (dp / 100) ** 2 # Prechareg scaling with SA
    neg_self_integral, pos_self_integral = ctm.integration_check(dp, charge_freq_den, v=False)
    self_integral = np.abs(neg_self_integral) + np.abs(pos_self_integral)
    pre_integral = ctm.trap_int(np.log10(dp), pre_charge_freq_den, np.min(np.log10(dp)), np.max(np.log10(dp)))
    time_fit = ctm.get_time_fit(dp, drop_height, p_p, p_f, g, mu, cfl, adjust_min_t, trace_time, convert_time)
    
    # charge_multiplier, precharge_ratio, time_shift = initial_guess[0], initial_guess[1], initial_guess[2]
    # time, selfcharge_trace, precharge_trace, total_trace = predict_trace(charge_multiplier, precharge_ratio, charge_freq_den, pre_charge_freq_den, self_integral, pre_integral, time_fit, sample_rate, trace_time)
    # total_residual, av_times, time, selfcharge_trace, precharge_trace, total_trace, R2 = cf.get_total_residual(time, total_trace, av_times, averaged_data, selfcharge_trace, precharge_trace, time_shift=time_shift, crop=True)
    # trace_plotter(time, total_trace, av_times, averaged_data, selfcharge_trace, precharge_trace)
    ###########################################################################
    # Fit

    constant_pcfd = pre_charge_freq_den.tolist() # Really weird fix, not sure why this gelps but code now works
    result = minimize(objective_function, initial_guess, args=(charge_freq_den, constant_pcfd, self_integral, pre_integral, time_fit, sample_rate, trace_time, av_times, averaged_data, minimise_op, adjust_by_mass))
    charge_multiplier, precharge_ratio, time_shift = result.x[0], result.x[1], result.x[2]
    
    print(f"Charge multiplier: {charge_multiplier}, precharge_ratio: {precharge_ratio}, time_shift: {time_shift}")
    print(f"[{charge_multiplier}, {precharge_ratio}, {time_shift}]")
    
    time, selfcharge_trace, precharge_trace, total_trace = predict_trace(charge_multiplier, precharge_ratio, charge_freq_den, pre_charge_freq_den, self_integral, pre_integral, time_fit, sample_rate, trace_time)
    total_residual, av_times, time, selfcharge_trace, precharge_trace, total_trace, R2 = cf.get_total_residual(time, total_trace, av_times, averaged_data, selfcharge_trace, precharge_trace, time_shift=time_shift, crop=True)
    trace_plotter(time, total_trace, av_times, averaged_data, selfcharge_trace, precharge_trace, average_minus_error=average_minus_error, average_plus_error=average_plus_error, title=None, error_lines=error_lines, adjust_by_mass=adjust_by_mass)

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


    
