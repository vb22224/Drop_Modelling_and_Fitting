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



def trace_plotter(time, total_trace, av_times, averaged_data, selfcharge_trace=[], precharge_trace=[], cutoff_time=10):
    
    """ Function to plot various modelled and measured traces """
    
    plt.figure(dpi=600)

    if selfcharge_trace is not None and len(selfcharge_trace) > 0:
        plt.plot(time, selfcharge_trace, '-', label='self-charging', linewidth=1.0) 

    if precharge_trace is not None and len(precharge_trace) > 0:
        plt.plot(time, precharge_trace, '-', label='pre-charging', linewidth=1.0) 
        
    plt.plot(time, total_trace, '-', label='predicted trace', linewidth=1.0) 
    plt.plot(av_times, averaged_data, '-', label='measured trace', linewidth=1.0) 
    plt.xlim(0, cutoff_time)
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
        for count, d in enumerate(total_trace):
            if d != 0:
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
    
    
    
    
if __name__ == "__main__":
    
    ###########################################################################
    # Parameters required for the Drop_Plotter
    
    path = "..\\Drops\\04_01_2024"
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
    store_dict, read_dict = False, True # Options to store or read from file the manual trim dictionary
    
    file_names, lol_structure, lol_labels = dp.get_file_names(input_type, path, file_names_lis, file_names_lol, remove_files, lol_labels)
    data, sample_rates, temperatures, humidities = dp.read_and_convert_data(file_names, path, plot_parameter, ignore_len_errors, trim, manual_trim, store_dict, read_dict)
    time_step, times, time = dp.get_times(sample_rates, data)
    av_times, averaged_data = dp.plot_figure(data, times, time, input_type, plot_parameter, file_names, path, lol_structure, lol_labels, change_type, plot_option, plot_average, show=False, get_av_data=True)
    
    ###########################################################################
    # Section for ring probe
    
    drop_height = 0.3725 # Height of particle drop / m
    target_cfl = 0.001
    p_p, p_f = 30, 1.23 # Density of particle and fluid in kg m^-3 (around 2000 for ash and 8940 for Cu)
    ring_height = drop_height / 3 # Height of the ring probe / m
    dp = 4610 # Particle size in um
    charge = -35 # Charge on particel in pC
    cutoff_time = 0.5
    
    time_lis, ring_trace = dtm.single_ring_trace(dp, charge, target_cfl=target_cfl, drop_height=drop_height, ring_height=ring_height, p_p=p_p, plot=False)
    time_lis, ring_trace = shift_prediction(time_lis, ring_trace, av_times, averaged_data)
    trace_plotter(time_lis, ring_trace, av_times, averaged_data, cutoff_time=cutoff_time)
    
    ###########################################################################
    # Combined
    
    # total_residual, av_times, time, selfcharge_trace, precharge_trace, total_trace = get_total_residual(time, total_trace, av_times, averaged_data, selfcharge_trace, precharge_trace, crop=True)
    # trace_plotter(time, total_trace, av_times, averaged_data, selfcharge_trace, precharge_trace)
    # print(f"The total residual for this fit is: {round(total_residual, 1)} pC")
    
    



    
