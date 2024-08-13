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
import statistics as stats



def get_size_charge(fit_type, dp, a, b, c, mode_sizes, mode_means, mode_stds, truncate=float('inf'), convert_size=False):
    
    """ Function to plot various modelled and measured traces """
    
    total_size = ctm.check_modes(mode_sizes, mode_means, mode_stds)
    frequency_density, charge_fit = [], []
    
    if fit_type == 'simple':
        for d in dp:
            charge_fit.append(ctm.calc_charge(d, a, b, c))
            frequency_density.append(ctm.calc_size_freq(d, mode_sizes, mode_means, mode_stds, total_size, truncate))
    elif fit_type == 'complex':
        for d in dp:
            charge_fit.append(ctm.calc_charge_complex(d, a, b))
            frequency_density.append(ctm.calc_size_freq(d, mode_sizes, mode_means, mode_stds, total_size, truncate))
    else:
        raise ValueError(f'fit_type should be "Simple" or "complex", instead: {fit_type}')

    frequency_density = np.array(frequency_density)
    charge_fit = np.array(charge_fit)

    if convert_size:
        vp = np.pi * np.power(np.array(dp), 3) / 6 # Volumes at given sizes
        frequency_density = frequency_density / vp # Converting volume distribution to number
        frequency_density /= ctm.trap_int(np.log10(np.array(dp)), frequency_density) # Normalisation

    return frequency_density, charge_fit



def trace_plotter(time, total_trace, av_times, averaged_data, selfcharge_trace=[], precharge_trace=[], adjust_by_mass=False, dpi=600):
    
    """ Function to plot various modelled and measured traces """
    
    plt.figure(dpi=dpi)

    if selfcharge_trace is not None and len(selfcharge_trace) > 0:
        plt.plot(time, selfcharge_trace, '-', label='self-charging', linewidth=1.0) 

    if precharge_trace is not None and len(precharge_trace) > 0:
        plt.plot(time, precharge_trace, '-', label='pre-charging', linewidth=1.0) 
        
        plt.plot([time[0], time[-1]], [0, 0], '--', linewidth=1.0, color='black')    
    plt.plot(time, total_trace, '-', label='predicted trace', linewidth=1.0) 
    plt.plot(av_times, averaged_data, '-', label='measured trace', linewidth=1.0)
    plt.xlim([time[0], time[-1]])
    plt.xlabel('Time / s')
    if adjust_by_mass:
        plt.ylabel('Specific Charge / pC g$^{-1}$')
    else:
        plt.ylabel('Charge / a.u. ')
    plt.legend() # loc='lower right'
    
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
    
    
    
def get_total_residual(time, total_trace, av_times, averaged_data, selfcharge_trace, precharge_trace, crop=True, time_shift=0, show_R2=False):
    
    """ Function to compute the total residual and R² value between the experimental and fitted trace """
    
    total_residual, SSE, SST = 0, 0, 0
    av_charge = stats.mean(averaged_data)
    
    if crop == True:
        
        if max(time) <= max(av_times):
            print("Error: the predicted timelength is less than the experimental time, so cannot crop")
        
        else:
            for number, t in enumerate(time): # Cropping the predicted trace
                if t >  max(av_times):
                    time = time[:number + 1]
                    total_trace = total_trace[:number + 1]
                    selfcharge_trace = selfcharge_trace[:number + 1]
                    precharge_trace = precharge_trace[:number + 1]
                    break

    if time_shift > 0: # Shifting trace left
        count_under = np.sum(time < time_shift) # Number of indecies to shift by
        total_trace = np.concatenate((np.full(count_under, total_trace[0]), total_trace[:-count_under]))
        selfcharge_trace = np.concatenate((np.full(count_under, selfcharge_trace[0]), selfcharge_trace[:-count_under]))
        precharge_trace = np.concatenate((np.full(count_under, precharge_trace[0]), precharge_trace[:-count_under]))
    else: # Shifting trace right
        count_under = np.sum(time < abs(time_shift)) # Number of indecies to shift by
        total_trace = np.concatenate((total_trace[count_under:], np.full(count_under, total_trace[-1])))
        selfcharge_trace = np.concatenate((selfcharge_trace[count_under:], np.full(count_under, selfcharge_trace[-1])))
        precharge_trace = np.concatenate((precharge_trace[count_under:], np.full(count_under, precharge_trace[-1])))

    for number, d in enumerate(averaged_data):
        index = np.abs(time - av_times[number]).argmin()
        residual = np.abs(d - total_trace[index])
        total_residual += residual
        SSE += residual ** 2
        SST += (d - av_charge) ** 2

    R2 = 1 - SSE / SST

    return total_residual, av_times, time, selfcharge_trace, precharge_trace, total_trace, R2
    

    
    
    
def predict_trace(charge_multiplier, precharge_ratio, charge_freq_den, pre_charge_freq_den, self_integral, pre_integral, time_fit, sample_rate, trace_time):
    
    """ Predicts the expected traces at specified precharge ratio and charge multiplier """
    
    pre_charge_freq_den *= precharge_ratio * self_integral / pre_integral
    time, selfcharge_trace = ctm.get_trace(charge_freq_den, time_fit, sample_rate, trace_time)
    time, precharge_trace = ctm.get_trace(pre_charge_freq_den, time_fit, sample_rate, trace_time)
    selfcharge_trace *= charge_multiplier
    precharge_trace *= charge_multiplier
    total_trace = selfcharge_trace + precharge_trace
    
    return time, selfcharge_trace, precharge_trace, total_trace
    
    
    
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
    file_names_lis = ['Lab2Small_1.txt', 'Lab2Small_2.txt', 'Lab2Small_3.txt', 'Lab2Small_4.txt']
    file_names_lol = [["Atitlan_1.txt", "Atitlan_2.txt", "Atitlan_3.txt", "Atitlan_4.txt", "Atitlan_5.txt"],
                      ["Atitlan_6.txt", "Atitlan_7.txt", "Atitlan_8.txt", "Atitlan_9.txt", "Atitlan_10.txt"]]
    remove_files = ['63-75_1.txt', '63-75_4.txt', '63-125_5.txt', '63-125_4.txt']
    lol_labels = ["Min", "Max"]
    spec_cat = ['CAlAin'] # specify categories to include if input type is categorised (file names before the underscore), leave empty to include all
    
    trim = True # If True removes the begining of the trace such that all traces start at the same time (required to calulate average trace)
    plot_average = True # If True plots the average trace instead of all individual traces
    error_lines = 'N' # Shows the error when averaged, should be area (A), line (L), or none (N)
    manual_trim = {"Cu 1 mm_3.txt": 2.25, "Cu 1 mm_5.txt": 2.95, "Cu 1 mm_6.txt": 3, "Cu 1 mm_7.txt": 4} # Dictionary of file names and the time (in seconds) that you want to maually trim from the start
    store_dict, read_dict = False, True # Options to store or read from file the manual trim dictionary
    adjust_by_mass = True
    base_time, tolerance_amp, tolerance_len = 0.5, 1.5, 300
    adjust_for_decay, time_const = True, 310.5 # in [s] and comes from the conductivity of air (David's = 310.5)
    data_col = 0 # Selects which column of data file to read (default is the first column = 0)
     
    file_names, lol_structure, lol_labels = dp.get_file_names(input_type, path, file_names_lis, file_names_lol, remove_files, lol_labels, spec_cat=spec_cat)
    data, sample_rates, temperatures, humidities, filtered_names = dp.read_and_convert_data(file_names, path, plot_parameter, ignore_len_errors, base_time, tolerance_amp, tolerance_len, trim=trim, manual_trim=manual_trim, store_dict=store_dict, read_dict=read_dict, data_col=data_col, adjust_by_mass=adjust_by_mass, adjust_for_decay=adjust_for_decay, time_const=time_const)
    time_step, times, time = dp.get_times(sample_rates, data)
    av_times, averaged_data = dp.plot_figure(data, times, time, input_type, plot_parameter, filtered_names, path, lol_structure, lol_labels, change_type, plot_option, plot_average, show=False, get_av_data=True, adjust_by_mass=adjust_by_mass)
    
    ###########################################################################
    # Parameters required for the Continuous_Trace_Model

    dp = np.logspace(np.log10(0.1), np.log10(10000), num=1000, base=10.0) # Evenly logspaced dp to veiw functions
    
    drop_height = 0.3725 # Height of particle drop / m
    g = 9.81 # Acceleration due to gravity in m s^-2
    p_f = 1.225 # Density of fluid in kg m^-3
    mu = 1.79E-5 # Viscosity of the fluid (air) in Pa s
    cfl = 1.0 # Taget CFL, ajust if numerical instabilities are encountered
    adjust_min_t = 0.1 # Parameter can be used to change the minimum timestep (default = 1)
    sample_rate = sample_rates[0] / 10 # Can devide this by less (or 1) to increse accracy but also time to compute
    
    initial_guess = [0.0034100625017912975, -11.888481910665362, -0.1] # Charge multiplier, precharge ratio, and time shift initial guesses
    charge_multiplier, precharge_ratio, time_shift = initial_guess
    
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

    frequency_density, charge_fit = get_size_charge(fit_type, dp, a, b, c, mode_sizes, mode_means, mode_stds, truncate, convert_size=convert_size)
    charge_freq_den = frequency_density * charge_fit
    pre_charge_freq_den = frequency_density * (dp / 100) ** 2 # Prechareg scaling with SA
    neg_self_integral, pos_self_integral = ctm.integration_check(dp, charge_freq_den, v=False)
    self_integral = np.abs(neg_self_integral) + np.abs(pos_self_integral)
    pre_integral = ctm.trap_int(np.log10(dp), pre_charge_freq_den, np.min(np.log10(dp)), np.max(np.log10(dp)))
    
    time_fit = ctm.get_time_fit(dp, drop_height, p_p, p_f, g, mu, cfl, adjust_min_t, trace_time, convert_time)

    time, selfcharge_trace, precharge_trace, total_trace = predict_trace(charge_multiplier, precharge_ratio, charge_freq_den, pre_charge_freq_den, self_integral, pre_integral, time_fit, sample_rate, trace_time)
    
    ###########################################################################
    # Combined Faraday Cup Section
    
    total_residual, av_times, time, selfcharge_trace, precharge_trace, total_trace, R2 = get_total_residual(time, total_trace, av_times, averaged_data, selfcharge_trace, precharge_trace, time_shift=time_shift, crop=trim)
    trace_plotter(time, total_trace, av_times, averaged_data, selfcharge_trace, precharge_trace, adjust_by_mass=adjust_by_mass)
    print(f"Total residual: {round(total_residual, 1)} pC, R² value: {round(R2, 3)}")
    
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


    
